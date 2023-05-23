import basic

while True:

    text = input('basic > ')
    result, error = basic.run('<stdin>', text)

    if error:

        print(error.asString())

    elif result:

        print(result)
