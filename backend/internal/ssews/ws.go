package ssews

import (
	"fmt"
	"sync"

	"github.com/gorilla/websocket"
)

type WSWriter struct {
	conn   *websocket.Conn
	mutex  sync.Mutex
	closed bool
}

func NewWSWriter(conn *websocket.Conn) *WSWriter {
	return &WSWriter{
		conn: conn,
	}
}

func (w *WSWriter) WriteEvent(event string, payload interface{}) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.closed {
		return fmt.Errorf("writer closed")
	}

	message := map[string]interface{}{
		"event": event,
		"data":  payload,
	}

	return w.conn.WriteJSON(message)
}

func (w *WSWriter) WriteError(code, message string) error {
	return w.WriteEvent("error", ErrorInfo{Code: code, Message: message})
}

func (w *WSWriter) Close() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.closed {
		return nil
	}

	w.closed = true
	return w.conn.Close()
}

func (w *WSWriter) ReadJSON(v interface{}) error {
	return w.conn.ReadJSON(v)
}

func (w *WSWriter) IsClosed() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	return w.closed
}
