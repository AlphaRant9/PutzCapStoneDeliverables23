import basic

while True:
    text = input('basic > ')
    
    result, error = basic.identifyTokens(text)

    # if error: print(error.as_string())
    # else: print(result)
