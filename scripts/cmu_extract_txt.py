import sys

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()
for line in lines:
    line = line[2:]
    fn = line[:line.find(' ')]
    text = line[line.find(' ') + 2:]
    text = text[:text.rfind("\" )")]
    print("'" + fn + "' '" + text + "'")
    f = open(fn + '.txt', 'w')
    f.write(text)
    f.close()
