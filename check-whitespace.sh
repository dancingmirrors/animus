#!/usr/bin/env bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

total_issues=0
files_with_issues=0

check_file() {
    local file="$1"
    local file_issues=0
    local output=""

    local whitespace_only=$(grep -n "^[[:space:]]\+$" "$file" 2>/dev/null | wc -l)
    if [ $whitespace_only -gt 0 ]; then
        output+="  ${YELLOW}Lines with only whitespace ($whitespace_only):${NC}\n"
        while IFS= read -r line; do
            output+="    Line $(echo $line | cut -d: -f1)\n"
        done < <(grep -n "^[[:space:]]\+$" "$file" | head -10)
        file_issues=$((file_issues + whitespace_only))
    fi

    local trailing_whitespace=$(grep -n "[^[:space:]][[:space:]]\+$" "$file" 2>/dev/null | wc -l)
    if [ $trailing_whitespace -gt 0 ]; then
        output+="  ${YELLOW}Lines ending with whitespace ($trailing_whitespace):${NC}\n"
        while IFS= read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            output+="    Line $line_num\n"
        done < <(grep -n "[^[:space:]][[:space:]]\+$" "$file" | head -10)
        file_issues=$((file_issues + trailing_whitespace))
    fi

    local equals_space=$(grep -n "=[[:space:]]\+$" "$file" 2>/dev/null | wc -l)
    if [ $equals_space -gt 0 ]; then
        output+="  ${YELLOW}Lines ending with '= ' ($equals_space):${NC}\n"
        while IFS= read -r line; do
            line_num=$(echo "$line" | cut -d: -f1)
            output+="    Line $line_num\n"
        done < <(grep -n "=[[:space:]]\+$" "$file" | head -10)
        file_issues=$((file_issues + equals_space))
    fi

    if [ $file_issues -gt 0 ]; then
        echo -e "${RED}File: $file${NC}"
        echo -e "$output"
        total_issues=$((total_issues + file_issues))
        files_with_issues=$((files_with_issues + 1))
    fi
}

echo "=== Checking .py files ==="
py_file_count=0
while IFS= read -r file; do
    check_file "$file"
    py_file_count=$((py_file_count + 1))
done < <(find . -maxdepth 1 -name "*.py" -type f | grep -v ".git" | grep -v "^\./build/")

echo "=== Summary ==="
echo "Python files checked: $py_file_count"
echo "Files with issues: $files_with_issues"
echo "Total issues found: $total_issues"

if [ $total_issues -eq 0 ]; then
    echo -e "${GREEN}✓ No whitespace issues found!${NC}"
    exit 0
else
    echo -e "${RED}✗ Found $total_issues whitespace issues in $files_with_issues files${NC}"
    exit 1
fi
