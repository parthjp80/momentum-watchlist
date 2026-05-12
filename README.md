# Momentum Stock Watchlist — TrueNAS Docker Deployment

Scores a configurable list of tickers across 7 momentum criteria, picks the
top 10, enriches them with Claude AI analysis, and emails a beautifully
formatted HTML report.

---

## How scoring works

| Criterion | Default weight | What it measures |
|---|---|---|
| Modified Sharpe ratio | 20% | Trend smoothness: mean return ÷ volatility (no risk-free rate), annualised |
| Volume analysis | 18% | Current volume vs 10-day average — spikes signal conviction |
| Price breakout | 15% | Distance above/below 20-day high; 52-week range position |
| EMA trend (9-period) | 15% | Price above/below rising 9 EMA; slope direction |
| ATR momentum | 12% | Current ATR vs 30-day ATR — expanding range = momentum |
| Tape reading | 10% | Proxy: volume surge + positive price action + EMA position |
| Technical pattern | 10% | 52-week percentile + recent directional consistency |

All weights are fully configurable via environment variables.

---

## Prerequisites

- TrueNAS SCALE (or any Docker host)
- Docker and Docker Compose installed
- An Anthropic API key — https://console.anthropic.com
- SMTP credentials (Gmail App Password, Outlook, or self-hosted mail)

---

## Deployment on TrueNAS SCALE

### 1  Transfer files

Copy the entire project folder to your TrueNAS dataset, e.g.:

```
/mnt/tank/watchlist/
```

You can use SCP, SFTP, or the TrueNAS web file manager.

### 2  Create your .env file

```bash
cd /mnt/tank/watchlist
cp .env.example .env
nano .env        # fill in API key, email settings, tickers
```

**Gmail users:** generate an App Password at
https://myaccount.google.com/apppasswords — do not use your real Gmail password.

### 3  Build the image

```bash
docker compose build
```

### 4  Test run (runs once immediately)

```bash
docker compose run --rm watchlist
```

Check your inbox. If it works, set up the daily schedule below.

### 5  Schedule daily emails via TrueNAS cron

In the TrueNAS web UI:

1. Go to **System → Advanced → Cron Jobs → Add**
2. Set the command:
   ```
   docker compose -f /mnt/tank/watchlist/docker-compose.yml run --rm watchlist
   ```
3. Set your schedule — recommended: **Monday–Friday at 7:00 AM** so you get
   the report before the US market opens at 9:30 AM ET.
4. Run as: `root` (or a user with Docker access)

Or add it directly to the system crontab via Shell:

```bash
# Edit cron
crontab -e

# Add this line (runs Mon-Fri at 7:00 AM):
0 7 * * 1-5 cd /mnt/tank/watchlist && docker compose run --rm watchlist >> /mnt/tank/watchlist/watchlist.log 2>&1
```

### 6  Optional: Docker Compose scheduler (no cron needed)

Uncomment the `scheduler` service in `docker-compose.yml` and create
`ofelia.ini` in the same folder:

```ini
[job-run "watchlist"]
schedule  = 0 7 * * 1-5
container = momentum_watchlist
```

Then run:
```bash
docker compose up -d scheduler
```

---

## Customising the watchlist

Edit the `WATCHLIST` variable in `.env`:

```
WATCHLIST=AAPL,MSFT,NVDA,AMZN,META,TSLA,PLTR,CRWD,SOFI,HOOD
```

Any valid US ticker symbol works. You can scan 10 or 100+ tickers —
the script rate-limits itself politely.

## Adjusting signal weights

Change the `W_*` variables in `.env`. They don't need to sum to exactly 100
(the code normalises them), but keeping them near 100 makes the percentages
intuitive:

```
W_SHARPE=25     # emphasise trend quality
W_VOLUME=25     # emphasise volume conviction
W_BREAKOUT=20
W_EMA=15
W_ATR=10
W_TAPE=5
W_PATTERN=0     # disable a criterion entirely
```

---

## Project structure

```
watchlist/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── app/
│   └── watchlist.py        # main script
└── templates/
    └── email.html          # Jinja2 HTML email template
```

---

## Troubleshooting

**"No module named yfinance"** — run `docker compose build` again to
reinstall dependencies.

**Email not arriving** — check spam folder; verify SMTP credentials with:
```bash
docker compose run --rm watchlist python -c "
import smtplib, os
s = smtplib.SMTP(os.environ['SMTP_HOST'], int(os.environ['SMTP_PORT']))
s.starttls(); s.login(os.environ['SMTP_USER'], os.environ['SMTP_PASS'])
print('SMTP OK')
"
```

**Gmail "less secure app" error** — you must use an App Password, not your
account password. Enable 2FA first, then create the App Password at
https://myaccount.google.com/apppasswords.

**yfinance rate limit** — if scanning 50+ tickers you may hit Yahoo rate
limits. Add `time.sleep(1)` in `watchlist.py` around the fetch loop, or
reduce the ticker list.

---

## Disclaimer

This tool is for informational purposes only and does not constitute financial
advice. Always conduct your own research before making investment decisions.
