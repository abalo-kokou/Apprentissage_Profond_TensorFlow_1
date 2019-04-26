import random

# Les bases a diviser
fin = open("./data/spam/spam.data", 'rb')
f75out = open("./data/spam/spam.trn", 'wb')
f25out = open("./data/spam/spam.tst", 'wb')

# fin = open("./data/ovarian/ovarian.data", 'rb')
# f75out = open("./data/ovarian/ovarian.trn", 'wb')
# f25out = open("./data/ovarian/ovarian.tst", 'wb')

for line in fin:
    r = random.random()
    if r < 0.75:
        f75out.write(line)
    else:
        f25out.write(line)
fin.close()
f75out.close()
f25out.close()