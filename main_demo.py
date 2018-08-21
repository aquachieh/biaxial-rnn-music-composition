'''
$ THEANO_FLAGS=device=gpu python main_demo.py [trad/xmax/children/jazz]
'''
import model
import main
import multi_training
import cPickle as pickle
import sys

WEIGHT_DIR = ".../good_model/"
music_style = sys.argv[1]

if music_style == "trad":
    WEIGHT_PATH = WEIGHT_DIR + "trad_params9500.p"
    #WEIGHT_PATH = WEIGHT_DIR + "trad_final_learned_config.p"
elif music_style == "xmax":
    WEIGHT_PATH = WEIGHT_DIR + "xmax_params9000.p"
elif music_style == "children":
    WEIGHT_PATH = WEIGHT_DIR + "children_params1000.p"
elif music_style == "jazz":
    WEIGHT_PATH = WEIGHT_DIR + "jazz_params19500.p "

print "\n=====  style:"+ music_style +"  =====\n"



pcs = multi_training.loadPieces("music_1")
m = model.Model([300,300],[100,50], dropout=0.5)

m.learned_config = pickle.load(open(WEIGHT_PATH, "rb" ) )
main.gen_adaptive(m,pcs,10,name="composition_1"+music_style+"_1")
main.gen_adaptive(m,pcs,10,name="composition_1"+music_style+"_2")
main.gen_adaptive(m,pcs,10,name="composition_1"+music_style+"_3")
main.gen_adaptive(m,pcs,10,name="composition_1"+music_style+"_4")
main.gen_adaptive(m,pcs,10,name="composition_1"+music_style+"_5")
