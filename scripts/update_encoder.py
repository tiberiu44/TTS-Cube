f_in = open("data/models/rnn_encoder.network-backup", "r")
f_out = open("data/models/rnn_encoder.network", "w")
add_params = False
for line in f_in.readlines():
    if not add_params:
        if line == '#LookupParameter# /_2 {200,19} 60801 ZERO_GRAD\n':
            line = '#LookupParameter# /_2 {200,20} 60801 ZERO_GRAD\n'
            add_params = True
            print("Found")
        f_out.write(line)
    else:
        line = line[:-1]
        f_out.write(line)
        parts = line.split(' ')
        for part in parts[-200:]:
            f_out.write(' ' + part)
        f_out.write('\n')
        add_params = False
f_out.close()
f_in.close()
