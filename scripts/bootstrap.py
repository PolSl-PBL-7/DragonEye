import subprocess
import sys
import os
import platform


BOOTSTRAP_FILE = os.path.join(os.getcwd(), 'scripts', 'bootstrap.toml')
OS = platform.system()
ARCH = platform.machine()


def exec(what: dict):
    for w in what:
        tmp = ''
        try:
            tmp = w['package'] + w['version']
        except KeyError:
            tmp = w['package']
        package_version = tmp.split()

        tmp = ''
        try:
            tmp = w['url']
            if tmp:
                tmp = f'-f {tmp}'
        except KeyError:
            pass
        url = tmp.split()

        tmp = ''
        try:
            tmp = w['command']
            if tmp.startswith('pip'):
                tmp = f'{sys.executable} -m {tmp}'
        except KeyError:
            tmp = f'{sys.executable} -m pip install'
        command = tmp.split()

        print(command, package_version)
        subprocess.check_call(args=command + package_version + url,
                              stderr=sys.stderr, stdout=sys.stdout, stdin=sys.stdin)


def bootstrap():
    subprocess.check_call(args=f'{sys.executable} -m pip install toml'.split(),
                          stderr=sys.stderr, stdout=sys.stdout, stdin=sys.stdin)

    import toml
    try:
        d = toml.load(BOOTSTRAP_FILE)
    except FileNotFoundError:
        print('Make sure you are in the root of project!')
        print(f'{BOOTSTRAP_FILE}')
        exit(2)

    try:
        what = d[OS][ARCH]
        if what == 'default':
            exec(d[what])
        else:
            exec(d[OS][ARCH])
    except KeyError:
        print(f'Platform {OS}/{ARCH} is not supported!')


if __name__ == '__main__':
    bootstrap()
