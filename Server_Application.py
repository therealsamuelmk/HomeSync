import RPi.GPIO as GPIO
import socket

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(7, GPIO.OUT)  

# Setup server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8888))  
server_socket.listen(1)

print("Server listening...")

while True:
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)
    
    while True:
        data = client_socket.recv(1024).decode()
        if not data:
            break
        
        if data == "on":
            GPIO.output(7, GPIO.HIGH)
            print("GPIO pin 7 turned ON")
        elif data == "off":
            GPIO.output(7, GPIO.LOW)
            print("GPIO pin 7 turned OFF")
        else:
            # Display the message received from the client
            print("Received message:", data)
    
    client_socket.close()
