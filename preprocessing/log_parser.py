from logparser.Drain import LogParser

input_dir = './logs_out/' # The input directory of log file
output_dir = 'result/'  # The output directory of parsing results
log_file = ['db', 'app', 'wp1', 'wp2', 'jupyter', 'server']  # The input log file name

#log_format = '<Computer> <Date> <Time> <PID> <Content>' # Define log format to split message fields
log_format = '<Computer> <Date> <Time> <Content>'

# Regular expression list for optional preprocessing (default: [])
regex = [
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)' # IP
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
for log in log_file:
    parser.parse(log)