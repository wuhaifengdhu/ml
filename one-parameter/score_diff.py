import sys

if len(sys.argv) < 2:
    print("Command sample: python score_diff.py correlation_id")
    exit(1)

correlation_id = sys.argv[1]
compare_result = []

with open(correlation_id + ".csv") as f:
    lines = f.readlines()
    if len(lines) != 3:
        print("Duplicated correlation id exist, please remove duplicated transaction, and manually rerun this scrip")
        exit(1)
    header = None
    rtcs = None
    incubator = None
    for line in lines:
        if line.startswith("service"):
            header = line.split("|")
        elif line.startswith("riskunifiedcomputeserv"):
            rtcs = line.split("|")
        elif line.startswith("incubatorcomputeserv"):
            incubator = line.split("|")
    if len(header) != len(rtcs) or len(rtcs) != len(incubator):
        print("Error, length not equal: header length %d, rtcs length %d, incubator length %d" % (len(header), len(rtcs), len(incubator)))
        exit(1)
    print("header length: %d, rtcs score %s incubator score %s" %(len(header), rtcs[1], incubator[1]))

    compare_result = [(header[i], rtcs[i], incubator[i]) for i in range(len(header)) if rtcs[i] != incubator[i]]
    print("Diff variable number: %d" % (len(compare_result) -1))

with open(correlation_id + "_diff.csv", "w") as f:
    for record in compare_result:
        f.write("%s, %s, %s\n" % record)
print("running success!")
