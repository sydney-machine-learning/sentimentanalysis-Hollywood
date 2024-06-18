## Task Description

1. Extract the final timestamp from each SRT file according to the given directory structure.
2. Convert the final timestamp into minutes.
3. Count the number of lines in each SRT file, removing timestamps and numbers.
4. Compile a list of the top 10 most frequently occurring words in each SRT file.

### Output Structure

The output should be structured as a dictionary with the following format:

```json
{
  "movie": "AllAboutEve",
  "year": 1950,
  "numberofwords": "10000",
  "time": "100 mins",
  "toptenwords": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
}
```

### Details

- **movie**: The movie title with all punctuation and symbols removed.
- **numberofwords**: The total word count.
- **toptenwords**: The list of the top 10 most frequently occurring words.

