#!/bin/bash
# SigmaGov Paper Bundle - Claim Verification Script

echo "=== SigmaGov Claim Verification ==="
echo ""

cd code/

echo "1. Checking for 'sorry' declarations..."
# Check for sorry not in comments (-- or /- -/)
SORRY_COUNT=$(grep -rE "^\s*sorry|[^-]sorry" *.lean 2>/dev/null | grep -v "/-" | grep -v -- "-- " | wc -l)
if [ "$SORRY_COUNT" -eq 0 ]; then
    echo "   ✅ PASS: Zero sorry declarations"
else
    echo "   ❌ FAIL: Found $SORRY_COUNT sorry declarations"
fi

echo ""
echo "2. Counting Lean files..."
FILE_COUNT=$(ls *.lean 2>/dev/null | wc -l)
echo "   Found $FILE_COUNT Lean files"
if [ "$FILE_COUNT" -ge 14 ]; then
    echo "   ✅ PASS: $FILE_COUNT files found"
else
    echo "   ⚠️  WARNING: Expected at least 14 files"
fi

echo ""
echo "3. Counting total lines..."
LINE_COUNT=$(wc -l *.lean 2>/dev/null | tail -1 | awk '{print $1}')
echo "   Total: $LINE_COUNT lines"
if [ "$LINE_COUNT" -gt 5000 ]; then
    echo "   ✅ PASS: ~5,500 lines as claimed"
else
    echo "   ⚠️  WARNING: Less than expected"
fi

echo ""
echo "4. Building project..."
if lake build 2>&1 | grep -q "Build completed successfully"; then
    echo "   ✅ PASS: Build succeeds"
else
    echo "   ❌ FAIL: Build failed"
fi

echo ""
echo "=== Verification Complete ==="
