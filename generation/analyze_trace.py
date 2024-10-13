import pstats

# Load the profiler output file
stats = pstats.Stats('profiler_output.prof')

# Sort the statistics by cumulative time
stats.sort_stats('cumulative')

# Print the top 10 lines of the statistics
stats.print_stats(20)