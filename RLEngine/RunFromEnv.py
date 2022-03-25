import os
import RLEngine

if __name__ == "__main__":
    params={}
    params['cmd']=os.getenv('BIM_COMM')
    params['mod'] = os.getenv('BIM_MOD')
    params['env'] = os.getenv('BIM_ENV')
    params['batch'] = os.getenv('BIM_BAT')
    params['worst'] = os.getenv('BIM_WOR')
    params['gamma'] = os.getenv('BIM_GAM')
    params['uid'] = os.getenv('BIM_UID')
    params['opt']=os.getenv('BIM_OPT')
    params['folder']=os.getenv('BIM_FOL')
    params['steps']=os.getenv('BIM_STE')
    RLEngine.main(params)
