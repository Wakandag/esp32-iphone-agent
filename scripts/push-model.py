"""Push the on-device GGUF into the installed app's Documents folder.

The app looks for `Documents/model.gguf` first, so the model survives reinstalls and the IPA
stays small. iOS jailbreak sshd drops long single streams, so the file goes up in verified
40 MB chunks that are joined on the phone.

    python3 scripts/push-model.py model.gguf
    python3 scripts/push-model.py model.gguf --host 10.0.0.188 --bundle com.screenagent.app
"""
import argparse
import os
import sys
import time

import paramiko

PARTS = "/var/mobile/parts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="local .gguf file")
    p.add_argument("--host", default="10.0.0.188")
    p.add_argument("--user", default="root")
    p.add_argument("--password", default=os.environ.get("IOS_PW", "alpine"))
    p.add_argument("--bundle", default="com.screenagent.app")
    p.add_argument("--chunk-mb", type=int, default=40)
    return p.parse_args()


class Phone:
    def __init__(self, args):
        self.args = args
        self.client = self.connect()

    def connect(self):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(self.args.host, 22, self.args.user, self.args.password, timeout=25,
                  look_for_keys=False, allow_agent=False)
        return c

    def reconnect(self):
        try:
            self.client.close()
        except Exception:
            pass
        time.sleep(2)
        self.client = self.connect()

    def sh(self, command, timeout=180):
        _i, out, _e = self.client.exec_command(command, timeout=timeout)
        text = out.read().decode("utf8", "replace").strip()
        return text, out.channel.recv_exit_status()


def container_documents(phone, bundle):
    prefs, _rc = phone.sh("ls /var/mobile/Containers/Data/Application/*/Library/Preferences/%s.plist 2>/dev/null | head -1" % bundle)
    if not prefs:
        sys.exit("app data container for %s not found (is the app installed and opened once?)" % bundle)
    return os.path.dirname(os.path.dirname(os.path.dirname(prefs))) + "/Documents"


def main():
    args = parse_args()
    size = os.path.getsize(args.model)
    chunk = args.chunk_mb * 1024 * 1024
    total = (size + chunk - 1) // chunk

    phone = Phone(args)
    documents = container_documents(phone, args.bundle)
    phone.sh("mkdir -p %s && rm -rf %s && mkdir -p %s" % (documents, PARTS, PARTS))
    print("pushing %s (%.0f MB) to %s/model.gguf in %d chunks" % (os.path.basename(args.model), size / 1048576.0, documents, total))

    started = time.time()
    with open(args.model, "rb") as handle:
        for index in range(total):
            block = handle.read(chunk)
            name = "%s/p%03d" % (PARTS, index)
            for attempt in range(3):
                try:
                    channel = phone.client.get_transport().open_session()
                    channel.exec_command("cat > %s" % name)
                    channel.sendall(block)
                    channel.shutdown_write()
                    while not channel.exit_status_ready():
                        time.sleep(0.05)
                    channel.recv_exit_status()
                    channel.close()
                    if phone.sh("wc -c < %s 2>/dev/null || echo 0" % name)[0] == str(len(block)):
                        print("  %d/%d (%.0fs)" % (index + 1, total, time.time() - started))
                        break
                except Exception as error:  # the jailbreak sshd drops big writes: reconnect and retry
                    print("  %d failed (%s), retry %d" % (index + 1, error, attempt + 1))
                    phone.reconnect()
            else:
                sys.exit("chunk %d never arrived" % (index + 1))

    target = documents + "/model.gguf"
    phone.sh("cat %s/p* > %s && rm -rf %s && chown -R mobile:mobile %s" % (PARTS, target, PARTS, documents))
    print(phone.sh("wc -c < %s" % target)[0] or "0", "bytes in", target)
    phone.client.close()


if __name__ == "__main__":
    main()
