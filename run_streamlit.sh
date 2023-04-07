
ps ax | grep "streamlit"
ps ax | grep "5000"

ps ax | grep 5000 | awk '{print $1}' | xargs kill -9

ps ax | grep "streamlit"
ps ax | grep "5000"

streamlit run home.py  --server.port 5000
