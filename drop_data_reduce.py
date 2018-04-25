import sys

fname = sys.argv[1]

entry_dict = {}

last_entry = ""
with open(fname,"r") as f:
	for line in f:
		line = line.strip()
		if line.find(",") == -1:
			entry_dict[last_entry] = entry_dict[last_entry] + "," + line	
			continue
		mkey = line.split(",")[0]
		last_entry = mkey
		entry_dict[mkey] = line

print_lines = [entry_dict[x] for x in entry_dict]

print_lines.sort()

for line in print_lines:
	print(line)

