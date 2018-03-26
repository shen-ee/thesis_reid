def format_id(id):
    if(id<10):
        return "000"+str(id)
    if(id<100):
        return "00"+str(id)
    if(id<1000):
        return "0"+str(id)


if __name__ == "__main__":
    print(format_id(10))