import RLEngine
import os
import configparser

if __name__ == "__main__":
    currDirname = os.path.dirname(__file__)
    path = os.path.join(currDirname, "uid.0/RLE.conf")
    print(str(path))
    config = configparser.RawConfigParser()
    config.read(path)
    params={}
    params['cmd'] = config.get('RUN CONFIG', 'command')
    params['mod'] = config.get('RUN CONFIG', 'mod')
    params['env'] = config.get('RUN CONFIG', 'env')
    params['batch'] = config.get('RUN CONFIG', 'batch')
    params['worst'] = config.get('RUN CONFIG', 'worst')
    params['gamma'] = config.get('RUN CONFIG', 'gamma')
    params['uid'] = config.get('RUN CONFIG', 'uid')
    params['opt']=config.get('RUN CONFIG', 'opt')
    params['folder']=config.get('RUN CONFIG', 'folder')
    params['steps']=config.get('RUN CONFIG', 'steps')
    RLEngine.main(params)
