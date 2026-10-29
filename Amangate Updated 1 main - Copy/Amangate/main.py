from app import app, init_db, start_visitor_cleanup
import chatbot
import recognition

if __name__ == "__main__":
    init_db()
    start_visitor_cleanup() 
    recognition.warm_up_model()
    chatbot.warm_up_async()   
    app.run(debug=False, port=8000, host='127.0.0.1', use_reloader=False, threaded=True)