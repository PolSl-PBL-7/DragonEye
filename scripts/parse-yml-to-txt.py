import yaml


if __name__ == '__main__':
    d = yaml.load(open('environment.yml'), Loader=yaml.FullLoader)
    for k in d['dependencies']:
        if type(k) == dict:
            for z in k['pip']:
                print(z)
