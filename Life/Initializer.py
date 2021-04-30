import random
x_size = 1000
y_size = 1000

with open('graph_initial.dat', 'w') as output:
    header = str(x_size) + '\n' + str(y_size) + '\n'
    output.write(header)
    for y in range(y_size):
        str_to_file = ''
        for x in range(x_size):
            str_to_file += str(random.randint(0, x_size) % 2) + ' '
        str_to_file += '\n'
        output.write(str_to_file)