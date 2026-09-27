function result = findCrossSplitMaximums(complexUnit, referenceIndices, queryIndices, blockSize)
% Maximum phase-invariant similarity for every query against references.
referenceIndices = double(referenceIndices(:));
queryIndices = double(queryIndices(:));
blockSize = floor(double(blockSize));
numberOfCaptures = size(complexUnit,1);
if isempty(referenceIndices)
    error('The reference-index list is empty.');
end
if any(referenceIndices < 1 | referenceIndices > numberOfCaptures)
    error('One or more reference indices are invalid.');
end
if any(queryIndices < 1 | queryIndices > numberOfCaptures)
    error('One or more query indices are invalid.');
end
if ~isscalar(blockSize) || ~isfinite(blockSize) || blockSize < 1
    error('blockSize must be a positive scalar.');
end
numberOfQueries = numel(queryIndices);
if numberOfQueries == 0
    result = struct('QueryDatasetRow',zeros(0,1), ...
        'ReferenceDatasetRow',zeros(0,1), ...
        'MaximumSimilarity',zeros(0,1,'single'));
    return;
end
maximumSimilarity = zeros(numberOfQueries,1,'single');
matchedReferencePosition = zeros(numberOfQueries,1);
referenceMatrix = complexUnit(referenceIndices,:);
numberOfBlocks = ceil(numberOfQueries / blockSize);
fprintf('  Reference captures: %d\n',numel(referenceIndices));
fprintf('  Query captures: %d\n',numberOfQueries);
fprintf('  Query blocks: %d\n',numberOfBlocks);
for blockIndex = 1:numberOfBlocks
    firstQueryPosition = (blockIndex-1)*blockSize + 1;
    lastQueryPosition = min(blockIndex*blockSize,numberOfQueries);
    currentQueryPositions = firstQueryPosition:lastQueryPosition;
    currentQueryIndices = queryIndices(currentQueryPositions);
    queryMatrix = complexUnit(currentQueryIndices,:);
    similarityMatrix = abs(referenceMatrix * queryMatrix');
    [blockMaximum,blockReferencePosition] = max(similarityMatrix,[],1);
    maximumSimilarity(currentQueryPositions) = single(min(blockMaximum(:),1));
    matchedReferencePosition(currentQueryPositions) = blockReferencePosition(:);
    if blockIndex == 1 || mod(blockIndex,10) == 0 || blockIndex == numberOfBlocks
        fprintf('  Similarity blocks: %d / %d | queries %d-%d\n', ...
            blockIndex,numberOfBlocks,firstQueryPosition,lastQueryPosition);
    end
end
matchedReferenceDatasetRow = referenceIndices(matchedReferencePosition);
result = struct();
result.QueryDatasetRow = queryIndices(:);
result.ReferenceDatasetRow = matchedReferenceDatasetRow(:);
result.MaximumSimilarity = maximumSimilarity(:);
end
