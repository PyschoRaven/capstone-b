from app import app, init_db, start_visitor_cleanup
import chatbot
import recognition

if __name__ == "__main__":
    init_db()
    start_visitor_cleanup()  # first pass runs immediately, then every minute
    recognition.warm_up_model()
    chatbot.warm_up_async()  # the assistant's model loads in the background
    # debug=False: the interactive Werkzeug debugger lets anyone who can reach
    # this port execute arbitrary code on an unhandled exception — fine to flip
    # on temporarily while developing, but should stay off once gate terminals
    # reach this server over the network instead of just 127.0.0.1.
    app.run(debug=False, port=8000, host='127.0.0.1', use_reloader=False, threaded=True)
