"""CSCA08 Assignment 1, Fall 2017
 I hereby agree that the work contained herein is solely my work and that I
 have not received any external help from my peers, nor have I used any
 resources not directly supplied by the course in order to complete this
 assignment. I have not looked at anyone else's solution, and no one has
 looked at mine. I understand that by adding my name to this file, I am
 making a formal declaration, and any subsequent discovery of plagiarism
 or other academic misconduct could result in a charge of perjury in
 addition to other charges under the academic code of conduct of the
 University of Toronto Scarborough Campus
 Name: Dittam Dey
 UtorID: deyditta
 Student Number: 1004432811
 Date: November/1/2017
"""


def pair_genes(gene1, gene2):
    '''(str, str) -> bool
    Given (gene1) and (gene2) which are strings that represent genes, will
    return True only if all nucleotides of one gene can pair with the
    corresponding nucleotides of another gene.
    REQ: gene1 and gene2 must be comprised of only 'A', 'T', 'G', 'C'
    REQ: gene1 and gene2 must be the same length
    REQ: gene1 and gene2 cannot be empty
    >>> pair_genes('TCAG','AGTC')
    True
    >>> pair_genes('TCAG','CTGA')
    True
    >>> pair_genes('TCAG','TTGA')
    False
    '''
    # if genes are not of smae length they are automatically unpairable
    if len(gene1) != len(gene2):
        pairable = False
    else:
        # create list with all valid nucleotide pairs
        possible_pairs = ['AT', 'GC', 'TA', 'CG', 'A*', '*A', 'T*', '*T',
                          'G*', '*G', 'C*', '*C']
        # initialize pair-validity variables for forward and reversed gene2
        valid_forward, valid_reversed = True, True
        index = 0
        # run loop as long as nucleotide pairs of the forward or revered genes
        # are valid or index reached end of genes
        while (index < len(gene1)) and (valid_forward or valid_reversed):
            # Combine nucleotides of gene1 and gene2 at current index
            pair_forward = gene1[index] + gene2[index]
            # Check if if nucleotide pair AT CURRENT index is valid IFF
            # nucleotide pairs UP UNTIL current index are valid
            if valid_forward and (pair_forward in possible_pairs):
                valid_forward = True
            else:
                valid_forward = False

            # Combine nucleotides of gene1 and reversed gene2 at current index
            pair_reversed = gene1[index] + gene2[-(index + 1)]
            # Check if nucleotide pair of REVERSED gene2 at CURRENT index is
            # valid IFF nucleotide pairs in REVERSED gene2 UP UNTIL current
            # index are valid
            if valid_reversed and (pair_reversed in possible_pairs):
                valid_reversed = True
            else:
                valid_reversed = False
            # Continue to next index
            index += 1
        # Return whether all nucleotides in gene1 and gene2 can be paired
        # successfully, forward or reversed
        pairable = bool(max(valid_forward, valid_reversed))

    return pairable


def zip_length(gene):
    '''(str) -> int
    Given (gene) a string that represents a gene, will return the zip length;
    the number of consecutively valid nucleotide pairs a gene forms with itself
    starting from the first and last indices and moving inward.
    REQ: gene must be comprised of only 'A', 'T', 'G', 'C'
    REQ: gene cannot be empty
    >>>AGTCTCGCT
    2
    '''
    # create list with all valid nucleoride pairs
    possible_pairs = ['AT', 'TA', 'GC', 'CG']
    # initial variables for zip-length and validity of nucleotide pairs
    zip_length, valid_pairs = 0, True
    index = 0
    # run loop as long as index is less than half gene length and nucleotide
    # pairs are valid
    while (index < len(gene) // 2) and valid_pairs:
        # Combine nucleotides starting with first and last indexes and moving
        # inward as index increases
        pairs = gene[index] + gene[-(index + 1)]
        # check if nucleotide pair is valid
        if pairs in possible_pairs:
            # add 1 to zip length of gene
            zip_length += 1
        else:
            valid_pairs = False
        # continue to next index
        index += 1
    # return the total zip length of the gene
    return zip_length


def find_anchors(gene, ancr_start, ancr_end):
    '''(list, str, str) -> tuple or NoneType
    Given (gene), a list representation of a gene sequence with each
    element being a nucleotide, & (anchor_start) (anchor_end),
    two strings that represent the start/end anchors neccessary for splicing.
    Will return a tuple containg the indexes of the first occurrence of
    ancr_start & ancr_end in gene iff both anchors are found and
    ancr_start appears before ancr_end.
    REQ: gene, ancr_start, ancr_end can only be comprised of 'A', 'T', 'G', 'C'
    REQ: gene cannot be empty
    REQ: ancr_start, ancr_end must be shorter than gene
    >>> find_anchors(list('ATGACTTACGTATGTAC'), 'AC', 'GT')
    (3, 9)
    >>> find_anchors(list('ATGATTTAGATGA'), 'AC', 'GT')
    (-1, -1)
    '''
    # Initialize index variables for start and end anchors
    # -1 means no anchor found
    start_idx, end_idx = -1, -1
    idx = 0
    # Run loop until end of gene or until both start & end anchors found
    while idx < len(gene) and (-1 in (start_idx, end_idx)):
        # Splice gene from the start of current idx to the length of ancr_start
        current_squence = gene[idx:(idx + len(ancr_start))]
        # Iff ancr_strt not found, check if current_sequence matches ancr_start
        if (start_idx == -1) and (current_squence == list(ancr_start)):
            start_idx = idx

        # Splice gene from the start of current idx to the length of ancr_end
        current_squence = gene[idx:(idx + len(ancr_end))]
        # Iff ancr_end not found, check if current_sequence matches ancr_end
        if (end_idx == -1) and (current_squence == list(ancr_end)):
            end_idx = idx

        idx += 1
    # Return anchor indexes iff start_idx appears before end_idx
    if start_idx < end_idx:
        return (start_idx, end_idx)


def splice_gene(source, destination, ancr_st, ancr_end):
    '''(list, list, str, str) -> NoneType
    Given(source) & (destination), two list representations of genes with each
    element being a nucleotide of the gene & (anchor_st) & (anchor_end),
    two string sequences that represent the start/end end points of a splice.
    Will mutate source and destination such that the sequences between the
    anchors of the source gene is extracted and inserted between the anchors of
    the destination gene iff anchors are present & in order in both genes.
    REQ: source, desitnation, anchor_ stt, anchor_end must consist of only
    REQ: 'A', 'T', 'G', 'C'
    REQ: In source & desitnation, each element must represent one nucleotide
    >>> source, destination = list('ATGCCAGTCA'), list('CTGCATAGTAC')
    >>> splice_gene(source, destination, 'GC', 'GT')
    >>> destination == list('CTGCCAGTAC')
    True
    >>> source, destination = list('ATGCCAGTCA'), list('CTGCATAGTAC')
    >>> splice_gene(source, destination, 'AA', 'GG')
    >>> destination == list('CTGCATAGTAC')
    True
    '''
    # Find indexes of anchors in source & destination reading genes forward
    src_anchor_idx = find_anchors(source, ancr_st, ancr_end)
    dest_anchor_idx = find_anchors(destination, ancr_st, ancr_end)
    # Check if anchors are found in both destination & source
    if (dest_anchor_idx is not None) and (src_anchor_idx is not None):
        # Get the start/end anchor indexes of source & destination and
        # convert them to be relative to the genes presented.
        src_strt = src_anchor_idx[0]
        src_end = src_anchor_idx[1] + len(ancr_end)
        dest_strt = dest_anchor_idx[0]
        dest_end = dest_anchor_idx[1] + len(ancr_end)
        # Delete sequence between anchors in destination gene
        del destination[dest_strt:dest_end]
        # Insert the sequcence found btwn anchors from source gene in between
        # anchors in destination gene
        destination[dest_strt:dest_strt] = source[src_strt:src_end]
        # Remove the sequence including the anchors from source
        del source[src_strt:src_end]

    # Check if anchors in source are not found, but anchors in destination
    # are found
    elif (src_anchor_idx is None) and (isinstance(dest_anchor_idx, tuple)):
        # Find indexes of anchors in revese of source gene
        src_anchor_idx = find_anchors(source[::-1], ancr_st, ancr_end)
        # Check again if anchors are found in source gene
        if (src_anchor_idx is not None):
            # Get the start/end anchor indexes of source & destination and
            # convert them to be relative to the genes presented.
            src_strt = len(source) - 1 - src_anchor_idx[0]
            src_end = len(source) - src_anchor_idx[1] - len(ancr_end) - 1
            dest_strt = dest_anchor_idx[0]
            dest_end = dest_anchor_idx[1]
            # Delete sequence between anchors in destination gene
            del destination[dest_strt:dest_end + len(ancr_end)]
            # Insert the sequcence found btwn anchors from source gene in
            # between anchors in destination gene
            destination[dest_strt:dest_strt] = source[src_strt:src_end:-1]
            # Remove the sequence including the anchors from source
            del source[src_strt:src_end:-1]

    # Check if anchors in destination are not found, but anchors in source
    # are found
    elif (dest_anchor_idx is None) and (isinstance(src_anchor_idx, tuple)):
        # Find indexes of anchors in revese of destination gene
        dest_anchor_idx = find_anchors(destination[::-1], ancr_st, ancr_end)
        # Check again if anchors are found in destination gene
        if (dest_anchor_idx is not None):
            # get length of destination and source before mutation
            dest_len = len(destination)
            src_len = len(source)
            # Get the start/end anchor indexes of source & destination and
            # convert them to be relative to the genes presented.
            src_strt = src_len - src_anchor_idx[0]
            src_end = src_anchor_idx[1]
            dest_strt = dest_anchor_idx[0]
            dest_end = dest_len - dest_anchor_idx[1] - len(ancr_end)
            # Delete sequence between anchors in destination gene
            del destination[dest_len - dest_strt - 1:dest_end - 1:-1]
            # Insert the sequcence found btwn anchors from source gene in
            # between anchors in destination gene
            destination[dest_end:dest_end] = source[src_strt: src_len -
                                                    src_end - len(ancr_end):-1]
            # Remove the sequence including the anchors from source
            del source[src_strt: src_len - src_end - len(ancr_end):-1]
    # Check if anchors not found in both source and destination genes
    elif (dest_anchor_idx is None) and (src_anchor_idx is None):
        # Find indexes of anchors in revese of destination & source genes
        dest_anchor_idx = find_anchors(destination[::-1], ancr_st, ancr_end)
        src_anchor_idx = find_anchors(source[::-1], ancr_st, ancr_end)
        # Check again if anchors are found in destination & source gene
        if (dest_anchor_idx is not None) and (src_anchor_idx is not None):
            # get length of destination and source before mutation
            dest_len = len(destination)
            src_len = len(source)
            # Get the start/end anchor indexes of source & destination and
            # convert them to be relative to the genes presented.
            src_strt = src_anchor_idx[0]
            src_end = src_len - src_anchor_idx[1] - len(ancr_st)
            dest_strt = dest_anchor_idx[0]
            dest_end = dest_len - dest_anchor_idx[1] - len(ancr_end)
            # Delete sequence between anchors in destination gene
            del destination[dest_end: dest_len - dest_strt]
            # Insert the sequcence found btwn anchors from source gene in
            # between anchors in destination gene
            destination[dest_end:dest_end] = source[src_end:src_len - src_strt]
            # Remove the sequence including the anchors from source
            del source[src_end:src_len - src_strt]


def match_mask(gene, mask):
    '''(str,str) -> int
    Given (gene), a string representation of a gene sequence and (mask),
    a string representation of sequence that pairs with sections of a gene
    in order find a specific pattern. mask can contain multis which are denoted
    with square brackets, they can act as multiple nucleotides.
    Will return the starting index of the mask for the first matching sequence
    found in the gene. If no match found then it will return -1.
    REQ: gene and lines in file must contain only 'A', 'T', 'G', 'C'
    REQ: mask must contain only 'A', 'T', 'G', 'C', '[', ']', '*', '1'...'9'
    REQ: mask can not contain empty multis e.g. 'AT[]GC'
    REQ: mask can not start with a number
    REQ: len(mask) <= len(gene)
    >>> match_mask('TTGCCTAAACC','CG2[AC]T3')
    2
    >>> match_mask('AACCATAAA', 'G2[AT]*')
    2
    >>> match_mask('TTTTTTTTTTTTTTT', 'CG2[AC]T3')
    -1
    '''
    # Create a list with all valid nucelotide pairs
    possible_pairs = ['AT', 'GC', 'TA', 'CG', 'A*', '*A', 'T*', '*T',
                      'G*', '*G', 'C*', '*C']
    # ---Expand out the numbers in mask---
    # Initialize empty string for expanded version of mask
    expand_mask = ''
    for idx in range(len(mask)):
        # look for number in the mask
        if mask[idx].isdigit():
            # multiply the previous nucleotide by the number and add it to
            # expand_mask
            expand_mask += mask[idx - 1] * (int(mask[idx]) - 1)
        # for all other characters that are not numbers just add to expand_mask
        else:
            expand_mask += mask[idx]

    # ---Find length of mask---
    # initialize length of mask and index for iterating through mask
    mask_len, idx = 0, 0
    while idx < len(expand_mask):
        # look for multis in mask and do not add to length until end of multi
        if expand_mask[idx] == '[':
            while expand_mask[idx] != ']':
                idx += 1
            # multis count as 1 nucleotide, add 1 to mask_len
            mask_len += 1
        # For all other characters that are not multis add 1 to mask_len
        else:
            mask_len += 1
        idx += 1

    # ---Search for matching sequence---

    # initialize index for gene iteration and validity of sequence
    idx, sequence_valid = 0, False
    # run loop as long a match isn't found or reached end of gene
    while idx < len(gene) and not sequence_valid:
        # from idx create a slice of gene the length of mask to check if it
        # pairs with mask
        current_sequence = gene[idx:idx + mask_len]
        sequence_valid = True
        # initialize 2 counter variables for mask and current_sequence
        mask_cnt, sequ_cnt = 0, 0
        # Iterate through the mask as long as match not found or reached end
        while sequ_cnt < len(current_sequence) and sequence_valid:
            # If a multi is encountered, check if at least one nucleotide in
            # the multi pairs with current nucleotide in the sequence
            if expand_mask[sequ_cnt] == '[':
                # initialize list to store booleans for pair validity
                multi_valid = []
                # run loop until the end of the multi
                while expand_mask[sequ_cnt] != ']':
                    # check pair validity
                    if expand_mask[sequ_cnt] + \
                            current_sequence[mask_cnt] in possible_pairs:
                        multi_valid.append(True)
                    else:
                        multi_valid.append(False)
                    sequ_cnt += 1
                # If at least one valid pair in multi then classify multi as
                # valid
                if True in multi_valid:
                    sequence_valid = True
                else:
                    sequence_valid = False
                sequ_cnt += 1
                mask_cnt += 1
            # For all other characters that are not multis
            else:
                # Check if nucleotide pair is valid
                if expand_mask[sequ_cnt] + \
                        current_sequence[mask_cnt] in possible_pairs:
                    sequence_valid = True
                else:
                    sequence_valid = False
                sequ_cnt += 1
                mask_cnt += 1
        idx += 1
    # if idx reached end of gene with no matches found return -1
    if idx == len(gene):
        match_idx = -1
    # if match found return the starting index of the sequence in the gene
    else:
        match_idx = idx - 1
    return match_idx


def process_gene_file(file_handle, input_gene, mask):
    '''io.TextIOWrapper -> tuple
    Given a file handle that points to a file with 1 gene per line, and given
    input_gene & mask string representations of a gene and mask.
    Will return a tuple (p,m,z) where p is line number of the first gene in the
    file that pairs with input_gene followed by m the line number first gene
    that contains a sequence that matches the mask and finally z the
    longest zip length of a gene up until p or m which ever comes after.
    REQ: gene must contain only 'A', 'T', 'G', 'C'
    REQ: mask must contain only 'A', 'T', 'G', 'C', '[', ']', '*', '1'...'9'
    REQ: mask can not contain empty multis e.g. 'AT[]GC'
    REQ: mask can not start with a number
    REQ: len(mask) <= len(gene)
    '''
    # get first line file and strip the \n character
    line = file_handle.readline().strip('\n')
    # initialize gene_pair, mask_match and zip_len which represents
    # p,m,z respectively
    gene_pair = -1
    mask_match = -1
    zip_len = 0
    # initialize current zip lengthe as a comparison variable to zip_len
    current_zip_len = 0
    # idx is a counter for line number
    idx = 0
    # run loop until both a gene_pair and mask_match is found or end of file
    while (line != '') and ((gene_pair == -1) or (mask_match == -1)):
        # iff gene_pair not found check if current line has valid gene
        if (gene_pair == -1) and pair_genes(line, input_gene):
            gene_pair = idx
        # iff mask_match no found check if current line has valid gene
        if (mask_match == -1) and (match_mask(line, mask) != -1):
            mask_match = idx
        # get zip length of current gene in line
        current_zip_len = zip_length(line)
        # check if current gene has larger zip length
        if current_zip_len > zip_len:
            # set new largest zip length
            zip_len = current_zip_len
        # move to next line and strip \n character from line
        line = file_handle.readline().strip('\n')
        idx += 1
    return (gene_pair, mask_match, zip_len)
