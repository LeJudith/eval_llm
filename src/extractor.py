import spacy
from spacy.matcher import PhraseMatcher
from torch import nn


class KeywordExtractor(nn.Module):
    """
    Extract keywords from biomedical sentences using spaCy's PhraseMatcher.

    Parameters
    ----------
    keywords: list of reference keywords

    Inputs
    ----------
    sent: input sentence(s) as type string

    Outputs
    ----------
    matches: list of matched keywords

    Examples
    ----------
    keywords = ["lymphoplasma", "inflammatory"]
    text = "A dense lymphoplasma cellular inflammatory infiltrate..."
    extractor = KeywordExtractor(keywords)
    extractor(text)
    """

    def __init__(self, keywords, negation_patterns=None):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Convert keywords to spaCy Docs and add them to the matcher
        patterns = [self.nlp.make_doc(keyword) for keyword in keywords]
        self.matcher.add("KeywordList", patterns)

        # Default negation patterns for multi-word negations up to three words before a keyword
        if negation_patterns is None:
            negation_patterns = [
                ["geen"],  # Single word negation
                ["niet"],
                ["zonder"],
                ["afwezig"],
                ["geen", "teken", "van"],  # Multi-word pattern
                ["geen", "aanwijzingen", "voor"],
            ]

        # Flatten negation patterns to strings for easier checking
        self.negation_patterns = [" ".join(pattern) for pattern in negation_patterns]

    def forward(self, input):
        """
        :param sent: sentences as type string
        """
        # doc = self.nlp(input)  # Process the sentence with spaCy

        # # Find matches
        # matches = self.matcher(doc)
        # matched_keywords = [doc[start:end].text for match_id, start, end in matches]

        # # Deduplicate matches and return them
        # return list(set(matched_keywords))
        doc = self.nlp(input)  # Process the sentence with spaCy

        # Find matches
        matches = self.matcher(doc)
        matched_keywords = []
        for match_id, start, end in matches:
            match_text = doc[start:end].text

            # Look back up to three tokens for negation patterns
            for i in range(1, 4):
                if start - i >= 0:
                    preceding_tokens = " ".join(
                        [doc[start - j].text.lower() for j in range(i, 0, -1)]
                    )

                    if any(
                        preceding_tokens.startswith(pattern)
                        for pattern in self.negation_patterns
                    ):
                        match_text = f"{preceding_tokens} {match_text}"  # Combine negation pattern with keyword
                        break  # Stop if a pattern is found

            matched_keywords.append(match_text)
        lst = list(set(matched_keywords))
        lst_filtered = filter_specific_keywords(lst)
        # Deduplicate matches and return them
        return lst_filtered


def filter_specific_keywords(keywords):
    """
    Filter a list of keywords to remove any that are substrings of other keywords.

    Args:
        keywords (List[str]): A list of keywords to filter.
    Returns:
        List[str]: The filtered list of keywords.

    Example: exlcudes 'basaalcelcarcinoom' when 'nodulair basaalcelcarcinoom' is included
    """

    # Sort by length in descending order (longer phrases are more specific)
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    final_keywords = []

    for keyword in sorted_keywords:
        # Add the keyword if it is not a substring of any already included keywords
        if not any(keyword in included for included in final_keywords):
            final_keywords.append(keyword)

    return final_keywords
