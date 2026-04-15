cd /home/chendawww/workspace/rl-navibot/
# FastAPI模块解耦前可以直接这样
# python /home/chendawww/workspace/rl-navibot/app/main.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload