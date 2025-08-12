for obj in $(find .git/objects -type f -newermt "1 hour ago" | grep -v "pack"); do
  hash=$(echo $obj | sed 's/\.git\/objects\///' | sed 's/\///')
  type=$(git cat-file -t $hash)
  if [ "$type" = "blob" ]; then
    # check if this blob is reachable from any commit
    if ! git log --all --find-object=$hash | grep -q commit; then
      echo "Unreachable Blob: $hash"
      git cat-file -p $hash | head -n 3
      echo "---"
    fi
  fi
done
