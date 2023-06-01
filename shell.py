import basic

while True:

    text = input('basic > ')
    if text.strip() == "":
        continue
    result, error = basic.run('<stdin>', text)

    if error:
        print(error.asString())
    elif result:
        if len(repr(result.elements)) == 1:
            print(repr(result.elements[0]))
        else:
            print(repr(result))