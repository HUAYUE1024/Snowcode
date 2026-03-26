# codechat Agent 实用案例

## 案例 1：接手新项目 — 快速了解架构

刚 clone 一个项目，5 分钟搞懂整体结构：

```bash
# 1. 建索引
codechat ingest

# 2. 项目概览
codechat summary

# 3. 核心入口在哪？
codechat ask "这个项目的入口文件是哪个？main函数在哪里？"

# 4. 数据流怎么走？
codechat agent "用户请求从进入到返回的完整数据流是什么？"
```

**输出效果：**
```
Step 1 → list_dir    了解项目结构
Step 2 → search      找到 main/入口
Step 3 → read_file   读取入口文件
Step 4 → search      追踪路由/处理函数

Answer:
数据流: main.py:12 → Router.dispatch() → Handler.process() → Response
```

---

## 案例 2：排查 Bug — 追踪调用链

线上报错，只知道出问题的函数名，需要追踪完整调用链：

```bash
# 从报错函数往上追
codechat trace "process_payment"

# 往下看这个函数调用了什么
codechat agent "process_payment 函数内部调用了哪些其他函数？每个调用的参数和返回值是什么？"

# 搜相关的异常处理
codechat find "PaymentError|payment.*exception|raise.*payment"
```

---

## 案例 3：Code Review — 提交前自查

写完代码准备提 PR，先让 AI 审一遍：

```bash
# 全项目审查
codechat review

# 只审查你改的文件
codechat review src/auth/middleware.py

# 专门找安全问题
codechat agent "这个项目有没有 SQL 注入、路径遍历、敏感信息泄露的风险？逐个文件检查"

# 找 TODO/FIXME/HACK
codechat find "TODO|FIXME|HACK|XXX|TEMP"
```

---

## 案例 4：重构准备 — 分析依赖关系

想重构一个模块，先搞清楚谁依赖它：

```bash
# 谁调用了这个函数？
codechat agent "所有调用 UserService.create 的地方，列出每个调用点的文件和行号"

# 这个类被哪些文件继承/使用？
codechat find "extends BaseHandler|import BaseHandler"

# 对比重构前后的两个实现
codechat compare src/old_auth.py src/new_auth.py
```

---

## 案例 5：写测试 — 找边界条件

给某个模块写单元测试：

```bash
# 分析函数签名和边界
codechat explain "chunk_file"

# 生成测试用例建议
codechat test-suggest "VectorStore.query"

# 找这个模块的输入来源（可能的边界值）
codechat agent "chunk_file 的参数来源是什么？哪些地方调用了它？参数可能有哪些边界值？"
```

**输出效果：**
```
chunk_file 在 3 处被调用:
1. cli.py:116 — 传入文件内容，可能为空/超长/二进制
2. agent.py:230 — ReadMultipleTool 调用
3. tests/test_chunker.py — 测试调用

建议测试用例:
- 空文件 → 应返回 []
- 超大文件 → 应分块
- 只有注释的文件
- 混合语言（Python + SQL）
```

---

## 案例 6：学习开源项目 — 理解设计模式

读一个用了设计模式的项目：

```bash
# 找设计模式
codechat agent "这个项目用了哪些设计模式？（单例、工厂、观察者、策略等）每个模式在哪个文件"

# 找抽象层/接口
codechat find "class.*ABC|class.*Base|class.*Interface|class.*Protocol"

# 理解错误处理策略
codechat agent "这个项目的错误处理策略是什么？是集中处理还是分散在各处？用的是 exception 还是 error code？"
```

---

## 案例 7：性能分析 — 找瓶颈

程序跑得慢，找性能问题：

```bash
# 找循环和嵌套
codechat find "for.*for|while.*for"

# 找数据库查询
codechat find "\.query\(|\.execute\(|SELECT.*FROM|\.find\("

# 找没有缓存的重复计算
codechat agent "有没有在循环里重复计算相同结果的地方？有没有可以用缓存优化的热路径？"
```

---

## 案例 8：文档生成 — 补注释

代码缺注释，让 AI 帮忙补：

```bash
# 找没注释的函数
codechat find "^[ ]*def [a-z]"  # 找所有函数定义

# 让 AI 解释每个函数
codechat explain "scan_files"

# 生成 API 文档骨架
codechat agent "列出所有公开函数的签名和一句话描述，格式：- `函数名(参数)` — 用途"
```

---

## 案例 9：跨文件追踪 — 配置传播

某个配置值在多处使用，想搞清楚影响范围：

```bash
# 追踪环境变量使用
codechat find "DASHSCOPE_API_KEY|OPENAI_API_KEY|OLLAMA"

# 追踪配置读取
codechat agent "DASHSCOPE_API_KEY 在代码中的完整传播路径是什么？从读取到最终使用"

# 找硬编码的值
codechat find "\"localhost\"|\"127.0.0.1\"|\":8080\"|\":3000\""
```

---

## 案例 10：增量开发 — 在现有代码上加功能

想加一个新功能，需要知道改哪些文件：

```bash
# 找类似功能的实现
codechat ask "现在是怎么处理 JSON 文件的？我想加一个 YAML 支持"

# 找需要改的入口点
codechat agent "如果要给这个项目加一个 export markdown 功能，需要改哪些文件？每个文件要改什么？"

# 找现有的扩展点
codechat find "register|plugin|extension|hook|callback"
```

---

## 快速参考

```bash
# 基础问答
codechat ask "问题"

# 深度探索（自动规划+工具调用+记忆）
codechat agent "问题"              # 默认 5 步
codechat agent "问题" -s 3         # 限制 3 步
codechat agent "问题" --no-plan    # 跳过规划

# 专业分析
codechat explain "目标"            # 解释
codechat review                    # 审查
codechat find "模式"               # 搜索
codechat summary                   # 架构
codechat trace "函数"              # 调用链
codechat compare A B               # 对比
codechat test-suggest "目标"       # 测试建议
```
