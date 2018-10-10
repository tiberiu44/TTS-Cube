import sys, os
import numpy

def read_embeddings_gensim(in_file):
    vocab = {}
    vectors = []
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load(in_file, mmap='r')
    for i, key in enumerate (wv.vocab.keys()):
        vocab[key] = wv[key]
    return vocab

def main(in_folder, out_folder):
    we = read_embeddings_gensim("letter_vectors.wv")
    we["-"] = numpy.array([0e-5]*100)
    we["START"] = numpy.array([0e-4]*100)
    we["STOP"] = numpy.array([0e-4]*100)
    if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    for fi in os.listdir(in_folder):
        
        if fi.endswith('.txt'):
            fout = open(out_folder+'/'+fi[:-4]+'.lab', 'w')
            print (fi)
            speaker = fi.split('_')[0]

            with open(in_folder+'/'+fi) as f:
                zz = [str("{:.6f}".format(we["START"][i])) for i in range(len(we["START"]))]
                fout.write("START %s LOWER %s\n" %(speaker, " ".join(zz)))
                for line in f.readlines():
                    line = line.replace("-", "")
                    line = line.replace(" ", "-")
                    for c in line.strip():
                      if c.isalpha() or c=="-":
                        if c.isupper():
                            case = "UPPER"
                        else:
                            case = "LOWER"

                        fout.write("%s %s %s " %(c.lower(), speaker, case))
                        zz = [str("{:.6f}".format(we[c.lower()][i])) for i in range(len(we[c.lower()]))]
                        fout.write("%s\n" %(" ".join(zz)))
                fout.write("STOP %s LOWER %s\n" %(speaker, " ".join(zz)))
                        #sys.exit(1)
            f.close()
            fout.close()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
