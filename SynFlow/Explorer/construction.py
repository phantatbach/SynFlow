"""Search parsed corpora for POS-anchored dependency constructions."""

import os
import re
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from SynFlow.const import DEFAULT_PATTERN, SENT_ID_PATTERN, VALID_FILLER_FORMATS
from SynFlow.utils import build_graph, format_filler


@dataclass(frozen=True)
class ConstructionStep:
    """
    One relation step in a construction branch.

    Attributes:
        relation: Directed dependency relation to traverse, including the
            ``chi_`` or ``pa_`` prefix produced by ``build_graph``.
        allowed_pos: POS tags allowed for the reached node. ``None`` means the
            step has no POS restriction, represented in the input template as
            an empty list: ``[]``.
    """

    relation: str
    allowed_pos: Optional[Set[str]] = None


ConstructionBranch = List[ConstructionStep]
BranchMatch = Tuple[List[Tuple[str, str, str, str]], str]


@dataclass(frozen=True)
class ConstructionTemplate:
    """
    Parsed construction template with one anchor POS and required branches.

    Attributes:
        anchor_pos: POS tag of the anchor node where each search starts.
        branches: Required dependency branches from the anchor. A sentence
            matches only when every branch can be found from the same anchor.
    """

    anchor_pos: str
    branches: List[ConstructionBranch]


def build_context_lookup(
    sent_tokens: List[str],
    pattern: re.Pattern,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Build a token-id to token, lemma, and POS lookup for one sentence.

    Args:
        sent_tokens: Parsed token lines for one sentence. Each line must match
            ``pattern`` and contain token, lemma, POS, ID, HEAD, and DEPREL
            fields.
        pattern: Regular expression used to parse corpus token lines.

    Returns:
        Dictionary keyed by token ID. Each value is a tuple of
        ``(token, lemma, pos)`` for that token.
    """
    id2context = {}
    for line in sent_tokens:
        match = pattern.match(line)
        if not match:
            continue
        token, lemma, pos, idx, _, _ = match.groups()
        id2context[idx] = (token, lemma, pos)
    return id2context


def _strip_group(text: str) -> str:
    """
    Remove one surrounding pair of parentheses from a template component.

    Args:
        text: Raw template component, such as ``"(VERB)"`` or
            ``"(chi_obj, [NOUN])"``.

    Returns:
        The component without surrounding parentheses when they are present.
    """
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _split_template_parts(construction_template: str) -> List[str]:
    """
    Split a construction template into anchor and branch components.

    Args:
        construction_template: Full construction template. Components are
            separated by ``&``. The first component is the anchor POS and all
            following components are required branches.

    Returns:
        Non-empty, stripped template components.
    """
    return [part.strip() for part in construction_template.split("&") if part.strip()]


def _parse_step(step_text: str) -> ConstructionStep:
    """
    Parse one fixed-format relation step.

    Args:
        step_text: Step text in the required format
            ``"(relation, [POS1, POS2])"``. Use an empty POS list
            ``"(relation, [])"`` when the step has no POS restriction.

    Returns:
        A ``ConstructionStep`` containing the relation and optional POS
        restriction.

    Raises:
        ValueError: If ``step_text`` does not include both a relation and a POS
        restriction list in the fixed input format.
    """
    raw_step_text = step_text.strip()
    step_text = _strip_group(raw_step_text)
    if "," not in step_text:
        raise ValueError(
            "Construction steps must use the fixed format '(relation, [POS...])'. "
            f"Use an empty list for no POS restriction, e.g. '(chi_obj, [])'. Got: {raw_step_text!r}"
        )

    relation, pos_text = step_text.split(",", 1)
    relation = relation.strip()
    pos_text = pos_text.strip()
    if not relation or not pos_text.startswith("[") or not pos_text.endswith("]"):
        raise ValueError(
            "Construction steps must use the fixed format '(relation, [POS...])'. "
            f"Got: {raw_step_text!r}"
        )
    allowed_pos = {
        pos.strip()
        for pos in pos_text[1:-1].split(",")
        if pos.strip()
    }
    return ConstructionStep(relation=relation, allowed_pos=allowed_pos or None)


def parse_construction_template(construction_template: str) -> ConstructionTemplate:
    """
    Parse a construction template into an anchor POS and required branches.

    Args:
        construction_template: Template string. The first component is the
            anchor POS in parentheses, for example ``"(VERB)"``. Each later
            component is a required branch. Every relation step must use the
            fixed format ``"(relation, [POS...])"``. Use ``[]`` when there is
            no POS restriction.

            Examples:
            ``"(VERB) & > (chi_nsubj, []) & > (chi_obj, [])"``
            ``"(VERB) & > (chi_nsubj, [NOUN, PRON, PROPN])"``

    Returns:
        Parsed construction template.

    Raises:
        ValueError: If the template is missing an anchor, has no branches, or
        contains a malformed relation step.
    """
    parts = _split_template_parts(construction_template)
    if len(parts) < 2:
        raise ValueError("construction_template must include an anchor POS and at least one relation branch.")

    anchor_pos = _strip_group(parts[0])
    branches: List[ConstructionBranch] = []
    for branch_text in parts[1:]:
        branch_text = branch_text.strip()
        if branch_text.startswith(">"):
            branch_text = branch_text[1:].strip()
        steps = [_parse_step(step) for step in branch_text.split(">") if step.strip()]
        if not steps:
            raise ValueError(f"Invalid empty construction branch: {branch_text!r}")
        branches.append(steps)

    return ConstructionTemplate(anchor_pos=anchor_pos, branches=branches)


def _branch_to_path(branch: ConstructionBranch) -> str:
    """
    Convert a branch to its relation-only path string.

    Args:
        branch: Parsed construction branch.

    Returns:
        Relation labels joined with ``" > "``.
    """
    return " > ".join(step.relation for step in branch)


def _format_anchor_filler(token: str, lemma: str, pos: str, filler_format: str) -> str:
    """
    Format the anchor node for the output ``sfillers`` column.

    Args:
        token: Surface form of the anchor token.
        lemma: Lemma of the anchor token.
        pos: POS tag of the anchor token.
        filler_format: Requested filler format. For ``*/deprel`` formats, the
            anchor has no dependency edge, so POS is used instead.

    Returns:
        Formatted anchor filler string.
    """
    if filler_format == "token/deprel":
        return f"{token}/{pos}"
    if filler_format == "lemma/deprel":
        return f"{lemma}/{pos}"
    return format_filler(token, lemma, pos, None, filler_format)


def find_by_construction_branch(
    graph: Dict[str, List[str]],
    id2context: Dict[str, Tuple[str, str, str]],
    id2deprel: Dict[Tuple[str, str], str],
    anchor_id: str,
    branch: ConstructionBranch,
    filler_format: str,
) -> List[BranchMatch]:
    """
    Find matching fillers for one branch from one anchor node.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``. Neighbor
            edges include both child-ward ``chi_`` and parent-ward ``pa_``
            labels.
        id2context: Mapping from token ID to ``(token, lemma, pos)``.
        id2deprel: Mapping from directed ``(source_id, target_id)`` edge to its
            dependency relation label.
        anchor_id: Token ID where this branch search starts.
        branch: Parsed branch to match from ``anchor_id``.
        filler_format: Requested output format for reached nodes.

    Returns:
        Matching branch realizations. Each result is
        ``([(node_id, filler, pos, path_to_node), ...], actual_path)`` where
        ``path_to_node`` is the relation path from anchor to that filler.
    """
    results: List[BranchMatch] = []

    def dfs(
        node: str,
        depth: int,
        seen: Set[str],
        path_rels: List[str],
        path_nodes: List[Tuple[str, str, str, str]],
    ) -> None:
        """
        Recursively match the remaining steps in one construction branch.

        Args:
            node: Current token ID in the traversal.
            depth: Number of branch steps already matched.
            seen: Token IDs already visited in this path; used to avoid cycles
                in the bidirectional graph.
            path_rels: Relation labels already matched.
            path_nodes: Matched nodes already reached, together with their
                formatted filler, POS tag, and relation path from the anchor.
        """
        if depth == len(branch):
            results.append((path_nodes, " > ".join(path_rels)))
            return

        step = branch[depth]
        for nb in graph.get(node, []):
            if nb in seen:
                continue
            lbl = id2deprel.get((node, nb))
            if lbl != step.relation:
                continue
            token, lemma, pos = id2context[nb]
            if step.allowed_pos is not None and pos not in step.allowed_pos:
                continue
            filler = format_filler(token, lemma, pos, lbl, filler_format)
            next_path_rels = path_rels + [lbl]
            dfs(
                nb,
                depth + 1,
                seen | {nb},
                next_path_rels,
                path_nodes + [(nb, filler, pos, " > ".join(next_path_rels))],
            )

    dfs(anchor_id, 0, {anchor_id}, [], [])
    return results


def _actual_next_relations(
    graph: Dict[str, List[str]],
    id2deprel: Dict[Tuple[str, str], str],
    node: str,
    parent_node: Optional[str],
) -> Set[str]:
    """
    Collect outgoing relation labels from a node, ignoring the edge back to its parent.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2deprel: Mapping from directed edge to dependency relation label.
        node: Token ID whose next outward relations should be collected.
        parent_node: Previous token ID on the matched construction path. This
            reverse edge is ignored because it points back into the already
            matched construction.

    Returns:
        Set of relation labels available from ``node`` away from ``parent_node``.
    """
    return {
        id2deprel[(node, nb)]
        for nb in graph.get(node, [])
        if nb != parent_node and id2deprel.get((node, nb))
    }


def _matches_exact_construction(
    graph: Dict[str, List[str]],
    id2deprel: Dict[Tuple[str, str], str],
    anchor_id: str,
    branch_match_group: Tuple[BranchMatch, ...],
) -> bool:
    """
    Check whether one group of branch matches exhausts the local construction tree.

    ``close`` mode uses this function to restrict both width and depth. Every
    matched node may only have the outgoing relation labels explicitly present
    in the selected construction branches. Leaf nodes in the template must be
    leaves relative to the outward construction search.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2deprel: Mapping from directed edge to dependency relation label.
        anchor_id: Token ID where construction search starts.
        branch_match_group: One selected match for each required branch.

    Returns:
        ``True`` when the selected branch matches exactly cover the outward
        construction under ``anchor_id``; otherwise ``False``.
    """
    allowed_relations_by_node: Dict[str, Set[str]] = {anchor_id: set()}
    parent_by_node: Dict[str, str] = {}

    for path_nodes, actual_path in branch_match_group:
        previous_node = anchor_id
        path_relations = actual_path.split(" > ") if actual_path else []
        for relation, (node_id, _, _, _) in zip(path_relations, path_nodes):
            allowed_relations_by_node.setdefault(previous_node, set()).add(relation)
            allowed_relations_by_node.setdefault(node_id, set())
            parent_by_node[node_id] = previous_node
            previous_node = node_id

    for node, allowed_relations in allowed_relations_by_node.items():
        actual_relations = _actual_next_relations(
            graph,
            id2deprel,
            node,
            parent_by_node.get(node),
        )
        if actual_relations != allowed_relations:
            return False

    return True


def _dedupe_branch_matches(branch_matches: List[BranchMatch]) -> List[BranchMatch]:
    """
    Remove duplicate branch matches while preserving order.

    Args:
        branch_matches: Branch matches selected for output.

    Returns:
        Branch matches with duplicate node/path realizations removed.
    """
    seen: Set[Tuple[Tuple[Tuple[str, str], ...], str]] = set()
    deduped: List[BranchMatch] = []
    for path_nodes, actual_path in branch_matches:
        key = (tuple((node_id, path_to_node) for node_id, _, _, path_to_node in path_nodes), actual_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path_nodes, actual_path))
    return deduped


def _find_all_unique_paths(
    graph: Dict[str, List[str]],
    id2deprel: Dict[Tuple[str, str], str],
    anchor_id: str,
    max_path_depth: int,
) -> Set[str]:
    """
    Find all unique relation paths from an anchor up to a maximum depth.

    Args:
        graph: Bidirectional dependency graph from ``build_graph``.
        id2deprel: Mapping from directed edge to dependency relation label.
        anchor_id: Token ID where path enumeration starts.
        max_path_depth: Maximum number of relation steps to include.

    Returns:
        Set of relation-only paths, each joined with ``" > "``.
    """
    out: Set[str] = set()

    def dfs(node: str, depth: int, seen: Set[str], rel_path: List[str]) -> None:
        """
        Recursively enumerate relation paths from the anchor.

        Args:
            node: Current token ID in the traversal.
            depth: Current path depth.
            seen: Token IDs already visited in this path.
            rel_path: Relation labels traversed so far.
        """
        if depth == max_path_depth:
            out.add(" > ".join(rel_path))
            return

        has_neighbor = False
        for nb in graph.get(node, []):
            if nb in seen:
                continue
            lbl = id2deprel.get((node, nb))
            if not lbl:
                continue
            has_neighbor = True
            dfs(nb, depth + 1, seen | {nb}, rel_path + [lbl])

        if not has_neighbor and rel_path:
            out.add(" > ".join(rel_path))

    dfs(anchor_id, 0, {anchor_id}, [])
    return out


def _process_file(
    args: Tuple[
        str,
        str,
        Optional[re.Pattern],
        str,
        str,
        str,
    ],
) -> List[Dict[str, object]]:
    """
    Process one corpus file for construction matches.

    Args:
        args: Tuple containing ``(corpus_folder, fname, pattern,
            construction_template, search_mode, filler_format)``. ``corpus_folder``
            is the period/subfolder path containing ``fname``. ``pattern`` may
            be ``None`` to use ``DEFAULT_PATTERN``.

    Returns:
        List of output rows. Each row has ``sentence_id``, ``sentence``,
        ``sfillers``, and ``path`` keys.
    """
    corpus_folder, fname, pattern, construction_template, search_mode, filler_format = args
    pattern = pattern or DEFAULT_PATTERN
    parsed_template = parse_construction_template(construction_template)
    results: List[Dict[str, object]] = []

    filepath = os.path.join(corpus_folder, fname)
    with open(filepath, encoding="utf8") as fh:
        sent_tokens: List[str] = []
        sent_forms: List[str] = []
        sent_id: Optional[str] = None
        has_anchor = False

        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("<s id"):
                sent_tokens = []
                sent_forms = []
                has_anchor = False
                match = SENT_ID_PATTERN.match(line)
                sent_id = match.group(1) if match else None

            elif line.startswith("</s>"):
                if sent_tokens and has_anchor:
                    id2wp, graph, id2deprel = build_graph(sent_tokens, pattern)
                    id2context = build_context_lookup(sent_tokens, pattern)
                    sentence_text = " ".join(sent_forms)
                    anchor_ids = [
                        idx
                        for idx, lemma_pos in id2wp.items()
                        if lemma_pos.rsplit("/", 1)[-1] == parsed_template.anchor_pos
                    ]

                    for anchor_id in anchor_ids:
                        token, lemma, pos = id2context[anchor_id]
                        branch_matches_by_branch: List[List[BranchMatch]] = []
                        found_all = True
                        for branch in parsed_template.branches:
                            matches = find_by_construction_branch(
                                graph,
                                id2context,
                                id2deprel,
                                anchor_id,
                                branch,
                                filler_format,
                            )
                            if not matches:
                                found_all = False
                                break
                            branch_matches_by_branch.append(matches)

                        if not found_all:
                            continue

                        if search_mode == "close":
                            valid_match_groups = [
                                match_group
                                for match_group in product(*branch_matches_by_branch)
                                if _matches_exact_construction(graph, id2deprel, anchor_id, match_group)
                            ]
                            if not valid_match_groups:
                                continue
                            branch_matches = _dedupe_branch_matches([
                                branch_match
                                for match_group in valid_match_groups
                                for branch_match in match_group
                            ])
                        elif search_mode == "closeh":
                            required_direct = {branch[0].relation for branch in parsed_template.branches}
                            actual_direct = {
                                id2deprel.get((anchor_id, nb))
                                for nb in graph.get(anchor_id, [])
                                if id2deprel.get((anchor_id, nb))
                            }
                            if actual_direct != required_direct:
                                continue
                            branch_matches = _dedupe_branch_matches([
                                branch_match
                                for matches in branch_matches_by_branch
                                for branch_match in matches
                            ])
                        else:
                            branch_matches = _dedupe_branch_matches([
                                branch_match
                                for matches in branch_matches_by_branch
                                for branch_match in matches
                            ])

                        anchor_filler = _format_anchor_filler(token, lemma, pos, filler_format)
                        results.append(
                            {
                                "sentence_id": sent_id,
                                "sentence": sentence_text,
                                "sfillers": [anchor_filler],
                                "path": parsed_template.anchor_pos,
                            }
                        )
                        for path_nodes, _ in branch_matches:
                            for _, filler, _, path_to_node in path_nodes:
                                results.append(
                                    {
                                        "sentence_id": sent_id,
                                        "sentence": sentence_text,
                                        "sfillers": [filler],
                                        "path": path_to_node,
                                    }
                                )

                sent_tokens = []
                sent_forms = []

            else:
                sent_tokens.append(line)
                match = pattern.match(line)
                if match:
                    sent_forms.append(match.group(1))
                    if match.group(3) == parsed_template.anchor_pos:
                        has_anchor = True

    return results


def construction_explorer(
    corpus_folder: str,
    construction_template: str,
    pattern: re.Pattern = None,
    search_mode: str = "open",
    filler_format: str = "lemma/pos",
    num_processes: int = max(1, cpu_count() - 1),
) -> pd.DataFrame:
    """
    Search a corpus for dependency constructions anchored by a POS tag.

    Args:
        corpus_folder: Root corpus folder. The function expects period or
            subcorpus directories inside this root, and each subdirectory may
            contain ``.txt`` or ``.conllu`` parsed files.
        construction_template: Construction pattern such as
            ``"(VERB) & > (chi_nsubj, []) & > (chi_obj, [])"``. The first
            parenthesized value is the anchor POS. Every relation step must
            include a POS restriction list: use ``[]`` for no restriction, or
            values such as ``[NOUN, PRON, PROPN]`` for restricted matching.
        pattern: Regex used to parse token lines. Defaults to
            ``DEFAULT_PATTERN``.
        search_mode: ``"open"``, ``"close"``, or ``"closeh"``. ``open`` accepts
            anchors that include at least the required branches. ``close``
            restricts both width and depth: every matched node may only have
            the outgoing relations explicitly listed in the template.
            ``closeh`` restricts horizontal width only at the anchor: the
            anchor's direct relation set must match exactly, while deeper
            specialization under those direct slots is allowed.
        filler_format: Format for context fillers. Must be one of
            ``VALID_FILLER_FORMATS``.
        num_processes: Number of worker processes. Use ``1`` for inline,
            deterministic execution during debugging.

    Returns:
        DataFrame with columns ``sentence_id``, ``sentence``, ``sfillers``, and
        ``path``. Anchor rows use the anchor POS as ``path``. Reached-node rows
        use the relation path from the anchor to that node.

    Raises:
        ValueError: If the corpus folder does not exist, ``search_mode`` is not
        supported, ``filler_format`` is invalid, or the construction template is
        malformed.
    """
    if not os.path.isdir(corpus_folder):
        raise ValueError(f"Corpus folder does not exist: {corpus_folder}")
    if search_mode not in {"open", "close", "closeh"}:
        raise ValueError("search_mode must be one of: open, close, closeh")
    if filler_format not in VALID_FILLER_FORMATS:
        valid_formats = ", ".join(sorted(VALID_FILLER_FORMATS))
        raise ValueError(f"filler_format must be one of: {valid_formats}")

    pattern = pattern or DEFAULT_PATTERN
    num_procs = max(1, num_processes)
    rows: List[Dict[str, object]] = []

    for subfolder in os.listdir(corpus_folder):
        subfolder_path = os.path.join(corpus_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        files = [
            fname
            for fname in os.listdir(subfolder_path)
            if fname.endswith((".conllu", ".txt"))
        ]
        args = [
            (subfolder_path, fname, pattern, construction_template, search_mode, filler_format)
            for fname in files
        ]
        if num_procs == 1:
            file_results = (_process_file(arg) for arg in args)
        else:
            pool = Pool(num_procs)
            file_results = pool.imap_unordered(_process_file, args, chunksize=10)
        try:
            for file_rows in file_results:
                rows.extend(file_rows)
        finally:
            if num_procs != 1:
                pool.close()
                pool.join()

    return pd.DataFrame(rows, columns=["sentence_id", "sentence", "sfillers", "path"])
