import RPi.GPIO as GPIO
import socket

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(22, GPIO.OUT) 
GPIO.setup(23, GPIO.OUT) 
GPIO.setup(24, GPIO.OUT) 

# Setup server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.160.0.196', 8888))  
server_socket.listen(1)

print("Server listening...")

while True:
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)
    
    while True:
        data = client_socket.recv(1024).decode()
        if not data:
            break
        
        if "Lights On" in data:
            GPIO.output(21, GPIO.HIGH)
            print("Lights turned ON")
        elif "Lights Off" in data:
            GPIO.output(21, GPIO.LOW)
            print("Lights turned OFF")
        elif "Aircon Off" in data:
            GPIO.output(22, GPIO.LOW)
            print("Aircon turned OFF")
        elif "Aircon On" in data:
            GPIO.output(22, GPIO.HIGH)
            print("Aircon turned ON")
        elif "Water Off" in data:
            GPIO.output(23, GPIO.LOW)
            print("Pump turned OFF")
        elif "Water On" in data:
            GPIO.output(23, GPIO.HIGH)
            print("Water turned ON")
        else:
            # Display the message received from the client
            print("Received message:", data)
    
    client_socket.close()
