#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tree flags from gbpTrees."""


class TreeFlags:
    """
    Parse tree flags from gbpCode.

    Parameters
    ----------
    header_file : str
        Path to the `tree_flags.h` header file in the Meraxes codebase.

    Attributes
    ----------
    flags : dict
        The tree flags parsed from the tree_flags.h header file in Meraxes.
    """

    __header_file = "/home/smutch/models/21cm_sam/meraxes/src/tree_flags.h"

    def __init__(self, header_file=__header_file):
        self.flags = {}
        with open(header_file, "r") as fd:
            for line in fd:
                line = line.split()
                if len(line) == 0:
                    continue
                if not line[0].startswith("//") and not line[1].startswith("TTTP"):
                    self.flags[line[1]] = 2 ** int(line[2][4:])

    def parse(self, num):
        """Parse a number as a combination of gbpTrees flags and return the string
        representation.

        Parameters
        ----------
        num : int

        Returns
        -------
        flags : str
            The parsed flags separated by '|'.
        """
        match = []
        for s, v in list(self.flags.items()):
            if (num & v) == v:
                match.append(s)
        return "|".join(match)
