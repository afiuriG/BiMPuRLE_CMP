import RLEngine
import sys




if __name__ == "__main__":
    command='optimize'
    print('Little code to run sequentially several classes of RLE')
    print('----From this are messages coming from IZH RS RLE----')
    params={'mod':'IZH','env':'MouCarCon','batch':'5','worst':'5','gamma':'1.0','uid':'0'}
    params['opt']='RS'
    params['cmd']=command
    params['folder']='all'
    params['steps']='5000'
    #RLEngine.main(params)
    print('----From this are messages coming from IZH GA RLE----')
    params={'mod':'IZH','env':'MouCarCon','batch':'5','worst':'5','gamma':'1.0','uid':'0'}
    params['opt']='GA'
    params['cmd']=command
    params['folder']='all'
    params['steps']='80'
    #RLEngine.main(params)
    print('-----From this are messages coming from IZH BO RLE----')
    params={'mod':'IZH','env':'MouCarCon','batch':'5','worst':'5','gamma':'1.0','uid':'0'}
    params['opt']='BO'
    params['cmd']=command
    params['folder']='all'
    params['steps']='5'
    RLEngine.main(params)



