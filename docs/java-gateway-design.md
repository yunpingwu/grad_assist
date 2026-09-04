# Java 业务网关 + SSE 透传实现方案

> 状态: 设计中(未实现) · 日期: 2026-09-04（v0.1）
> 目标: 新增 Spring Boot 业务网关，承接**用户账号（邮箱+密码）、模型配置与 api\_key 加密存储**，并将前端 AI 请求以 **SSE 流式透传**到下沉后的 Python AI 服务。Python FastAPI 不再对外暴露，仅接受本网关注入的内部身份与已解析模型配置。

***

## 1. 背景与决策

| 决策项           | 结论                                                |
| ------------- | ------------------------------------------------- |
| 网关技术栈         | Spring Boot 3 + Spring Security + WebFlux（SSE 透传） |
| 登录形态          | 邮箱 + 密码（**不用 OAuth**），后端签发 JWT                    |
| 用户/模型存储       | MySQL（关系库）                                        |
| Python 侧定位    | 下沉为内部 AI 服务，仅受本网关注入身份                             |
| 会话 checkpoint | 仍留 MongoDB，隔离 key 由匿名 `user_id` 改为账号 `user.id`    |
| api\_key 存储   | MySQL 密文（AES-GCM），主密钥仅存环境变量                       |

***

## 2. 用户表设计（账号密码）

### 2.1 DDL

```sql
CREATE TABLE users (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  email         VARCHAR(255) NOT NULL,
  password_hash VARCHAR(100) NOT NULL,   -- bcrypt（60 字符）/ argon2
  nickname      VARCHAR(64),
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_email (email)
);
```

### 2.2 密码哈希

- 采用 **BCrypt**（Spring Security 自带 `BCryptPasswordEncoder`，cost=10\~12），避免自写盐/哈希逻辑。

- 数据库只存 `password_hash`，**永不存明文，也永不回传给前端**。

- 注册/改密时 `hash = encoder.encode(rawPassword)`；登录时 `encoder.matches(rawPassword, hash)`。

### 2.3 注册 / 登录 / JWT 流程

```
注册:  POST /api/auth/register  { email, password, nickname? }
       → 校验 email 唯一 → BCrypt 哈希 → 写 users → 签发 JWT

登录:  POST /api/auth/login     { email, password }
       → 查 users 校验哈希 → 签发 JWT

鉴权:  后续请求带 Authorization: Bearer <jwt>
       → Spring Security 过滤器本地验签 → 从 claims 解出 user_id 注入上下文
```

JWT 建议：HS256（单机共享密钥）或 RS256（未来多实例），`user_id`、`email` 入 claims，过期时间 7\~30 天 + refresh 可选。

***

## 3. 模型配置表

```sql
CREATE TABLE model_configs (
  id           BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id      BIGINT NOT NULL,
  label        VARCHAR(64)  NOT NULL,
  model        VARCHAR(128) NOT NULL,   -- 如 deepseek-v4-flash
  base_url     VARCHAR(255),
  api_key_enc  VARBINARY(512),          -- AES-GCM 密文
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_user (user_id),
  CONSTRAINT fk_model_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

加解密在应用层完成，禁止原样入库：

```
加密: api_key_enc = AES_GCM.encrypt(apiKey, MASTER_KEY)
解密: api_key     = AES_GCM.decrypt(api_key_enc, MASTER_KEY)
```

`MASTER_KEY` 只放环境变量；对外返回模型列表时 `api_key_enc` 一律**脱敏**（只回传有无配置，不回传密文给前端）。

***

## 4. 网关整体结构

```
com.gradassist.gateway
├── auth/       # 注册/登录、JWT 签发/校验、Security 过滤器
├── user/       # users 表 CRUD、/api/auth/me
├── model/      # model_configs CRUD、AES-GCM 加解密、本地缓存
├── proxy/      # SSE 透传到 Python（核心）
└── config/     # 数据源、WebClient 连接池、InternalToken、Python 地址
```

***

## 5. SSE 透传核心实现（边读边写）

### 5.1 为什么必须边读边写

Python `/study/chat` 是 `text/event-stream`：token 一帧一帧推送，最终 `done` 收尾。网关若把整个响应缓存完再转发，会导致：

- 前端打字机效果失效（整段一次性到达）

- 网关内存随长回答膨胀

- 首字延迟 = Python 全程耗时 + 缓冲耗时

正确做法是**逐块透传**：拿到上游 `Flux<DataBuffer>` 后直接写入下游响应，WebFlux 的响应式链天然提供背压与取消。

### 5.2 依赖

```xml
<!-- spring-boot-starter-webflux -->
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-security</artifactId>
</dependency>
<dependency>
  <groupId>io.jsonwebtoken</groupId>
  <artifactId>jjwt-api</artifactId>
</dependency>
```

### 5.3 转发端点

```java
@RestController
@RequestMapping("/api")
public class StudyProxy {

    private final WebClient pyClient;      // 指向 Python 内网地址
    private final ModelService modelService;

    @PostMapping("/study/chat")
    public Mono<ServerResponse> chat(ServerHttpRequest req) {
        String userId = AuthContext.requireUserId();   // 过滤器从 JWT 注入

        // 1. 读前端 JSON（只含 model_id，不含明文 key）
        // 2. 解析模型配置（本地缓存 + AES-GCM 解密），注入 Python 请求体
        // 3. 转发到 Python，拿到字节流 Flux，逐块透传
        return req.getBodyToMono(ChatBody.class)
            .flatMap(body -> {
                ModelConfig cfg = modelService.resolve(userId, body.modelId());
                return pyClient.post()
                    .uri("/study/chat")
                    .header("X-Internal-Token", INTERNAL_TOKEN)  // 内部信任
                    .header("X-User-Id", userId)
                    .bodyValue(toPythonBody(body, cfg))
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, resp -> Mono.empty()) // 非2xx也透传 body
                    .bodyToFlux(DataBuffer.class);   // 关键：不聚合，逐块流式
            })
            .flatMap(flux -> ServerResponse.ok()
                .contentType(MediaType.TEXT_EVENT_STREAM)
                .header(HttpHeaders.CACHE_CONTROL, "no-cache")
                .header("X-Accel-Buffering", "no")   // 让 nginx 等不缓冲
                .body(flux));                        // 直接绑定，边读边写
    }
}
```

### 5.4 响应头透传要点

- `Content-Type: text/event-stream`：前端按 SSE 解析

- `Cache-Control: no-cache`：禁止代理/浏览器缓冲

- `X-Accel-Buffering: no`：若前面有 nginx，禁用其缓冲

### 5.5 错误与中断处理

- Python 业务错误（教材未登记等）已在 SSE 流内以 `{"type":"error"}` 事件返回，**HTTP 仍是 200**，网关照常透传即可。

- 真正的 HTTP 4xx/5xx（极少）：`onStatus(isError, Mono.empty())` 忽略状态码、仍透传 body，避免网关吞掉错误上下文；若 body 为空则网关补发一个 `error` 事件。

### 5.6 背压与取消

- 上游（Python）慢 → `Flux` 慢消费 → 网关不堆内存（响应式背压）

- 浏览器断开 → 下游响应关闭 → 订阅被取消 → 自动取消对 Python 的上游订阅、释放连接

***

## 6. WebClient 连接池配置

```java
ConnectionProvider provider = ConnectionProvider.builder("py-pool")
    .maxConnections(500)
    .pendingAcquireMaxCount(1000)
    .pendingAcquireTimeout(Duration.ofSeconds(60))
    .maxIdleTime(Duration.ofSeconds(30))
    .maxLifeTime(Duration.ofSeconds(120))
    .build();

HttpClient httpClient = HttpClient.create(provider)
    .responseTimeout(Duration.ofMinutes(10))              // SSE 长连接，读超时要长
    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 3000)
    .option(ChannelOption.SO_KEEPALIVE, true);

WebClient pyClient = WebClient.builder()
    .baseUrl(PY_BASE_URL)
    .clientConnector(new ReactorClientHttpConnector(httpClient))
    .build();
```

要点：连接复用 + keep-alive，避免每轮 AI 对话新建 TCP；`responseTimeout` 需撑满一轮长对话（含工具调用）。

***

## 7. 鉴权过滤器与 model 注入

- **JWT 过滤器**：无状态本地验签，`user_id` 从 claims 解出、不查库；将 `user_id` 写入 `AuthContext`（WebFlux 用 `Reactor Context` 传递）。

- **model 解析**：`modelService.resolve(userId, modelId)` 走本地缓存（Caffeine，TTL 5\~10 min），命中不查 MySQL；未传 `modelId` 或模型属于他人则回落到「服务端默认模型」，不注入自定义配置。

- **内部信任**：Java→Python 附带 `X-Internal-Token`（环境变量共享），Python 侧新增中间件校验，防止绕过网关直连。

***

## 8. 性能与超时基准（预期）

| 环节               | 预期开销                       |
| ---------------- | -------------------------- |
| JWT 无状态验签        | < 1 ms                     |
| model 解析（缓存命中）   | < 1 ms                     |
| 网关↔Python 一跳（内网） | < 1\~2 ms                  |
| 单帧 SSE 转发        | < 1 ms                     |
| **一轮问答网关总增量**    | **< 几十 ms（占 AI 秒级耗时 <1%）** |

***

## 9. 落地要点与遗留

1. Python 侧改造点：`get_user_id` 依赖由 `X-User-Id 头` 改为「`X-Internal-Token` 校验 + 内部 `X-User-Id`」；`/study/chat` 的 `model/base_url/api_key` 改为由网关注入。
2. 前端侧：移除 `utils/models.ts` 中 api\_key 明文；模型列表改走 `GET /api/models`（脱敏）；登录页 + token 存储 + 请求头 `Authorization`。
3. Mongo 会话迁移：`study_checkpoints` 的 thread 从匿名 `user_id` 迁移绑定到账号 `users.id`（老匿名数据一次性归属策略待定）。
4. 密码找回/改密、refresh token、多端会话管理等作为后续扩展，不在 v0.1 范围。

