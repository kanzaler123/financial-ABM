# GitHub 展示层 · Financial ABM Observatory

这是独立于 `web/` 的只读研究展厅，不连接 FastAPI，不启动 Python 模拟，不改动正式验收协议，不解封任何种子。

## 预览

在项目根目录执行：

```sh
python showcase/build.py --output _site
python -m http.server 8080 --directory _site
```

访问 `http://localhost:8080`，也可以直接用浏览器打开生成的 `_site/index.html`。输出已内联 CSS、JavaScript 与数据，无 CDN、在线字体、遥测或外部图表库依赖。需要 Python 3.12+，构建只使用标准库，无须安装 ABM 或 npm 依赖。

## GitHub Pages

仓库启用 Pages 后，`GitHub showcase` 工作流会部署到 GitHub Pages。首次由仓库管理员在 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**，然后在 Actions 中运行 `GitHub showcase`。

未启用 Pages 时，CI 仍验证页面并提供名为 `github-showcase` 的离线展示产物；发布任务明确跳过，不伪报“已上线”。PR 只构建检查，不发布。

## 内容与数据约定

`build.py` 从三个已提交的 `mechanism_report.json` 读取原始数据，保留报告代码版本、SHA-256 和精确源文件链接。它不重算模拟。2026-08-14 正式归档为 18/22，开发与 holdout 各为 22/22；归档早于 2026-09-16 的代码审计。CI 通过不代表 Stage 1 完成。

分布图使用报告的跨种子中位数与 Q10–Q90，不是时序数据或置信区间。不补造证据：同名 evidence 缺失时明确提示查看原报告。示意图不表示投资者社交网络或真实资金流。

## 修改入口

| 文件 | 内容 |
|---|---|
| `index.html` | 页面语义结构与中文文案 |
| `styles.css` | 深色视觉系统、响应式与减少动效 |
| `app.js` | 机制链路、报告/指标切换、验收明细、数据导出 |
| `build.py` | 数据读取、来源验证、离线单页构建 |
| `test_showcase.py` | 数据与静态构建回归 |
| `../.github/assets/observatory.svg` | GitHub README 封面 |

```sh
python -m unittest discover -s showcase -p 'test_*.py'
node --check showcase/app.js
```

现有 `web/`、`src/abm/`、配置、报告和冻结清单均不由展示构建改写。后续 Codex 可独立优化本地 Web；两者只共享概念与报告，不共享会话或运行控制逻辑。
