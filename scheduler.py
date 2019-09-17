import sys
import json
import schedule
import time
import threading

reload(sys)
sys.setdefaultencoding('UTF8')

config_file = "load.conf"

instance_required_keys = ["instance_name", "schedule_pattern", "delay", "hive_table"]
config_required_keys = ["schedule_patterns", "instances_config"]
pattern_required_keys = ["mask"]


def write_config(config):
    with open(config_file, "w") as f:
        json.dump(config, f, indent=5)
        print("finished dump dic to file")


def load_config():
    with open(config_file, "r") as f:
        config = json.load(f)
        print("finished load config from json")
        return config


def validate_config(config_map, required_fields):
    for config_key in required_fields:
        if not config_map.has_key(config_key):
            print("Config key %s is required" % str(config_key))
            return False
    return True


def pattern_match(config_pattern, mask):
    return int("0b" + config_pattern, 2) & int("0b" + mask, 2) > 0


def schedule_job(instance_config, job_type):
    instance_name = instance_config.get(instance_required_keys[0])
    print("Active job type %s for instance %s" % (job_type, instance_name))
    if job_type == "secondly":
        schedule.every().second.do(job, instance_config, job_type)
    elif job_type == "minutely":
        schedule.every().minute.do(job, instance_config, job_type)
    elif job_type == "hourly":
        schedule.every().hour.do(job, instance_config, job_type)
    elif job_type == "daily":
        schedule.every().day.do(job, instance_config, job_type)
    elif job_type == "weekly":
        schedule.every(7).days.do(job, instance_config, job_type)
    else:
        print("Currently do not support schedule job type: %s" % job_type)


def start_schedule_job(instance_config, schedule_pattern):
    if not validate_config(instance_config, instance_required_keys):
        print("Instance config not passed validate: %s" % str(instance_config))
        return
    instance_pattern_config = instance_config.get(instance_required_keys[1])
    for pattern_type, pattern in schedule_pattern.items():
        if not validate_config(pattern, pattern_required_keys):
            print("Pattern config not passed validate: %s" % str(pattern))
        if pattern_match(instance_pattern_config, pattern.get(pattern_required_keys[0])):
            schedule_job(instance_config, pattern_type)


def job(instance_config, job_type):
    threading.Thread(target=job_task, args=(instance_config, job_type)).start()


def job_task(instance_config, job_type):
    instance_name = instance_config.get(instance_required_keys[0])
    print("Running job type %s on config %s" % (job_type, instance_name))


def run_main():
    config = load_config()
    if not validate_config(config, config_required_keys):
        print("Config not passed validate, please check config: %s" % str(config))
        return

    for instance_config in config.get(config_required_keys[1]):
        start_schedule_job(instance_config, config.get(config_required_keys[0]))

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    run_main()

