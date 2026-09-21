missing_files=""
for file in $(git ls-tree -r --name-only 9c6b2e5ddff3a042f5248e2641390274c1b3a644); do
  if [ ! -e "$file" ]; then
    missing_files="$missing_files $file"
  fi
done

if [ -n "$missing_files" ]; then
  echo "Restoring: $missing_files"
  git checkout 9c6b2e5ddff3a042f5248e2641390274c1b3a644 -- $missing_files
  git restore --staged $missing_files
else
  echo "No missing files to restore from 9c6b."
fi
