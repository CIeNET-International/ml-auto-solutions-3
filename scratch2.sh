for commit in 9c6b2e5ddff3a042f5248e2641390274c1b3a644 4082868d53dabe07418f4a55fcf6231831ef67d6 219dbfcf82f7d4a909bc8590630ce00480a328d9; do
  echo "Checking $commit"
  git ls-tree -r $commit | grep -i exist
done
