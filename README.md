# 跨频段回撤修复因子 · CFPR

一个可复现的 Python 量化研究项目，研究以下假设：**此前上涨、近期适度回调、今天尾盘放量修复的股票，随后五个交易日是否有更高收益？**

项目把低频背景与高频修复信号相乘，并提供数据校验、因子计算、Rank IC、控制变量回归、含交易成本的组合回测和研究报告。默认采用 A 股交易时段：收盘后计算信号，最早下一交易日开盘交易。

**目前没有真实行情回测结论。** 随附示例使用模拟数据，作用是验证计算与研究流程；它没有预设未来收益优势，不能据此判断因子有效。

## 快速运行

需要 Python 3.11 或更新版本，以及 [uv](https://docs.astral.sh/uv/)。在仓库根目录运行：

```bash
uv sync --locked --extra dev
uv run cfpr demo --output outputs/demo --symbols 40 --days 160 --seed 42
```

依赖安装完成后，示例无需行情账号或外部网络访问。它生成日频与尾盘分钟模拟数据，再完成整套研究流程。查看 `outputs/demo/report.md`、`summary.json` 与 `diagnostics.png`；逐日结果和交易记录也会保存为 CSV。

也可以用 pip 安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cfpr demo --output outputs/demo
```

`uv.lock` 用于复现锁定的依赖；pip 安装会按项目声明重新解析版本。

## 发布到 GitHub

项目已包含中文说明、MIT 许可证、锁定依赖和 GitHub Actions 测试。首次提交可运行：

```bash
git add .
git commit -m "Initial cross-frequency pullback-recovery research project"
git push -u origin HEAD:main
```

发布前核对 `origin` 指向自己的仓库。`.gitignore` 已排除原始行情、运行结果、虚拟环境和 `.env`，请勿把行情供应商密钥或受限数据提交到仓库。

## 因子定义

令 $[x]_+=\max(x,0)$，默认因子为：

$$
F_{i,t}=[M_{i,t}]_+\cdot D_{i,t}e^{-D_{i,t}^2/2}\cdot[H_{i,t}]_+\cdot\min(Q_{i,t},2)
$$

| 分量 | 默认定义 | 研究含义 |
| --- | --- | --- |
| $M$ | $t-25$ 至 $t-5$ 的收益率，除以日收益标准差乘 $\sqrt{20}$ | 回调前的上涨背景 |
| $D$ | $t-5$ 至 $t-1$ 的下跌幅度，除以日收益标准差乘 $\sqrt{4}$；未下跌取零 | 回调深度 |
| $H$ | 今天最后 60 分钟收益率，除以前 20 个交易日同时段收益率的标准差 | 尾盘修复强度 |
| $Q$ | 今天最后 60 分钟成交量，除以前 20 个交易日同时段平均成交量 | 尾盘相对成交量 |

日收益标准差取 $t-20$ 至 $t-1$ 的 20 个日收益，使用样本标准差。两个波动率基准和成交量基准均不包含今天。默认尾盘窗口为北京时间 14:00–15:00，分钟条以开始时间标记，即 14:00 至 14:59。

$g(D)=De^{-D^2/2}$ 在 $D=1$ 达到最大，表达“适度回调”的假设。缺少历史数据或波动率为零时，因子是缺失值；满足数据条件但缺少上涨、回调或修复时，因子为零。这两种情况分别处理。

完整时间轴、公式与统计口径见 [研究方法](docs/methodology.md)。

## 使用自己的行情

准备 CSV 或 Parquet 文件：

| 数据 | 必需字段 |
| --- | --- |
| 日频 | `date,symbol,open,close,volume`；可选 `tradable` |
| 一分钟行情 | `timestamp,symbol,open,close,volume` |
| 交易日历（建议提供） | `date` |

```bash
uv run cfpr run \
  --daily data/daily.parquet \
  --intraday data/intraday.parquet \
  --calendar data/calendar.csv \
  --output outputs/research \
  --horizon 5 \
  --top-fraction 0.2 \
  --cost-bps 10
```

分钟数据只需包含尾盘窗口，也可以提供全日行情。窗口必须完整；程序不会自动补造缺失分钟。价格需要使用一致的复权口径，股票池需要保留历史退市股票和当时可知的纳入条件。停牌、涨跌停与上市时间等交易约束应由数据明确表达。

文件示例、时区、复权、停牌及缺失值约定见 [数据说明](docs/data.md)。项目不绑定行情供应商，也不包含未经授权的真实市场数据。

## 研究结果如何解读

- **预测力：** 每个信号日计算 Rank IC 和分位数组合未来收益，零分样本仍参与有效截面；并与普通趋势、短期反转、日频相对成交量、单独尾盘修复信号比较。
- **增量信息：** 每日截面回归同时纳入因子、此前趋势、短期反转与日频相对成交量。汇总系数与 IC 使用 Newey–West 标准误，滞后阶数至少为 `horizon - 1`，处理五日标签的重叠。
- **可执行组合：** 默认每五个交易日建立一组等权多头持仓；目标持仓数为全部有效股票数的 20% 向上取整，再按得分选取至多该数量的正分股票。下一日开盘买入，第五日收盘卖出，各组不重叠。默认成本为买、卖各 10 bp。
- **同口径比较：** 基准按相同交易时点和成本，等权持有全部有效因子样本，包括零分股票。它是研究股票池基准，不是市场指数。

回测不实现融券做空。次日无法买入的仓位保留现金；若已持仓股票缺少估值价格或无法按计划卖出，程序报错，避免默默删除困难样本。更完整的 A 股涨跌停、停牌延期成交、佣金、印花税和冲击成本模型属于后续扩展。

统计显著、样本内收益或模拟运行成功，都不能单独证明策略具备实盘优势。应在固定参数后使用真实、时点一致的数据做样本外检验，并检查交易成本敏感性。

## 命令与输出

```bash
# 只生成可复现的模拟行情
uv run cfpr generate --output data/demo --symbols 40 --days 160 --seed 42

# 只计算因子，不读取未来收益
uv run cfpr factor --daily data/demo/daily.csv \
  --intraday data/demo/intraday.csv --output outputs/factors.csv

# 验证代码
uv run pytest
uv run ruff check .

# 查看完整参数
uv run cfpr --help
uv run cfpr run --help
```

| 输出 | 内容 |
| --- | --- |
| `factors.csv` | 分量、得分、有效性与缺失原因 |
| `forward_returns.csv` | 下一开盘至第五日收盘的标签、实际入场与离场日期 |
| `ic_daily.csv`、`baseline_ic.csv` | 因子及对照信号的逐日 Rank IC |
| `quantile_returns.csv` | 保留并列分数的分位数组合未来收益 |
| `regression_daily.csv` | 逐日增量回归系数、样本数与跳过原因 |
| `equity.csv`、`trades.csv` | 净值、逐日收益与交易明细 |
| `summary.json`、`report.md`、`diagnostics.png` | 汇总指标、可读报告与研究图表 |
| `config.json`、`manifest.json` | 本次参数与输入文件信息 |

项目结构：

```text
src/cfpr/
  config.py        参数与合法性检查
  data.py          CSV / Parquet、时区与完整性校验
  factor.py        多频段因子与对照信号
  synthetic.py     可复现模拟行情
  statistics.py    IC、分组与增量回归
  backtest.py      标签、交易成本与持仓净值
  reporting.py     研究报告与图表
  pipeline.py      端到端研究流程
  cli.py          命令行入口
tests/             因子时点、数据泄漏、成交和统计测试
docs/              方法与数据契约
```

## 下一步研究

1. 使用真实历史股票池，处理退市、ST、IPO、复权与分钟缺失问题。
2. 固定初始参数，按时间划分训练与样本外区间，避免反复调参后只展示最佳结果。
3. 比较仅高频、仅低频、加法组合与当前乘法交互，验证增量信息来自哪里。
4. 加入行业、市值与流动性控制，检验不同市场状态、持有期和成本下的稳定性。
5. 根据实际交易规则完善成交模拟，再讨论容量与可交易收益。

相关机制的背景阅读：[Federal Reserve Bank of New York, Staff Report No. 513](https://www.newyorkfed.org/research/staff_reports/sr513.html)。已有研究不等于证明本因子的有效性，也不构成对其新颖性的认证。

许可证：[MIT](LICENSE)。
