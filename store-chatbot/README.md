### create and activate venv
python3 -m venv venv
source venv/bin/activate

### run the app
python main.py

### Talk with the bot:
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "message": "I am not feeling well today."}'
