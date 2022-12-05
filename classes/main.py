print("Hello World")

with open("/config") as f:
    print(*f.readlines())

