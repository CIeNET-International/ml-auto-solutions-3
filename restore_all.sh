missing_files=""
for file in $(git ls-tree -r --name-only 4082868d53dabe07418f4a55fcf6231831ef67d6); do
  if [ ! -e "$file" ]; then
    missing_files="$missing_files $file"
  fi
done

if [ -n "$missing_files" ]; then
  echo "Restoring: $missing_files"
  git checkout 4082868d53dabe07418f4a55fcf6231831ef67d6 -- $missing_files
  git restore --staged $missing_files
else
  echo "No missing files to restore."
fi
