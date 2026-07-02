import subprocess
import tempfile
from pathlib import Path

import dendropy
import numpy as np
import torch
from Bio import AlignIO, SeqIO

dna = ["<pad>", "A", "C", "G", "T", "-"]
aa = [
    "<pad>",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "-",
]


def newick_to_distance_matrix(newick):
    tns = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns)
    pdm = tree.phylogenetic_distance_matrix()
    taxa = [t.label for t in tree.taxon_namespace]
    dist = np.array(
        [[pdm(t1, t2) for t2 in tree.taxon_namespace] for t1 in tree.taxon_namespace]
    )
    return dist, taxa


def simulate_stick_breaking(alpha, max_profiles=100):
    betas = np.random.beta(1, alpha, size=max_profiles)
    remaining_stick = np.cumprod(np.concatenate([[1], 1 - betas[:-1]]))
    weights = betas * remaining_stick
    K = np.searchsorted(np.cumsum(weights), 0.999) + 1
    weights = weights[:K].clip(min=1e-5)  # ignore 1/1000
    weights /= weights.sum()
    return weights


def write_nexus_profiles(weights, profiles, path):
    with open(path, "w") as f:
        f.write("#nexus\nbegin models;\n")
        for i, p in enumerate(profiles):
            freqs = "/".join(f"{x:.10f}" for x in p)
            f.write(f"frequency pi{i + 1} = {freqs};\n")
        f.write("end;\n")

    f_str = "{" + ",".join(f"pi{i + 1}:1:{w:.10f}" for i, w in enumerate(weights)) + "}"
    return f_str


def simulate_alignment(newick, cfg, D, seed=0):
    if cfg.mode == "cont":
        return simulate_cont(newick, cfg, D, seed=seed)
    elif cfg.mode == "dna" or cfg.mode == "aa":
        return simulate_iqtree(newick, cfg, D, seed=seed)


def align(seqs, taxa):
    """Align sequences using Kalign3. Returns aligned numpy array."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        for name, seq in zip(taxa, seqs):
            f.write(f">{name}\n{seq}\n")
        in_file = f.name

    out_file = in_file.replace(".fa", ".aln")
    subprocess.run(
        ["kalign", "-i", in_file, "-o", out_file],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    aln = AlignIO.read(out_file, "fasta")
    Path(in_file).unlink()
    Path(out_file).unlink()

    aln_dict = {rec.id: str(rec.seq) for rec in aln}
    if aln_dict:
        gap_seq = "-" * len(next(iter(aln_dict.values())))
    else:
        gap_seq = "-"
    return np.array([aln_dict.get(t, gap_seq) for t in taxa])


def simulate_cont(newick, cfg, D, seed=0):
    rng = np.random.default_rng(seed)
    dist, taxa = newick_to_distance_matrix(newick)
    N = len(taxa)
    d_lam, d_vec = np.linalg.eigh(np.exp(-0.5 * dist / cfg.hyper.lengthscale))
    d_lam = np.maximum(d_lam, 0 * d_lam)
    d_sqrt = d_vec @ ((d_lam**0.5)[:, None] * d_vec.T)
    eig_dir = d_sqrt @ rng.standard_normal((N, D))
    alignment = eig_dir @ rng.standard_normal((D, D)) / np.sqrt(D)
    return alignment, taxa


def simulate_iqtree(newick, cfg, D, seed=0):
    rng = np.random.default_rng(seed + 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        outprefix = tmpdir / "sim"

        tree_file = tmpdir / "tree.tre"
        tree_file.write_text(newick)

        cmd = [
            "iqtree3",
            "--alisim",
            str(outprefix),
            "-t",
            str(tree_file),
            "--seed",
            str(seed),
            "--length",
            str(D),
        ]

        if cfg.model == "JC":
            assert cfg.mode == "dna"
            model_str = "JC"
            alphabet = dna
            seq_type = "DNA"

        elif cfg.model == "GTR":
            assert cfg.mode == "dna"
            q_matrix = rng.exponential(size=6).clip(min=1e-5)
            q_matrix = (q_matrix[:5] / q_matrix[-1]).astype(str)
            q_str = "{" + ",".join(q_matrix) + "}"
            i_rate = cfg.hyper.max_invariant * rng.uniform()
            i_str = "{" + str(i_rate) + "}"
            g_rate = max(rng.exponential(), 1e-5)
            g_str = "{" + str(g_rate) + "}"
            model_str = f"GTR{q_str}+GC{g_str}+I{i_str}"
            alphabet = dna
            seq_type = "DNA"

        elif cfg.model == "C60":
            assert cfg.mode == "aa"
            i_rate = cfg.hyper.max_invariant * rng.uniform()
            i_str = "{" + str(i_rate) + "}"
            model_str = f"LG+C60+I{i_str}"
            alphabet = aa
            seq_type = "AA"

        elif cfg.model == "CAT":
            assert cfg.mode == "aa"
            q_matrix = rng.exponential(np.ones(20 * 19 // 2)).clip(min=1e-5)
            q_matrix = (q_matrix[:-1] / q_matrix[-1]).astype(str)
            q_str = "{" + ",".join(q_matrix) + "}"
            g_rate = max(rng.exponential(), 1e-5)
            g_str = "{" + str(g_rate) + "}"

            dispersity_parameter = 10 * max(rng.exponential(1), 1e-5)
            cat_probs = simulate_stick_breaking(dispersity_parameter, max_profiles=100)
            cat_pis = rng.dirichlet(
                rng.exponential(size=20).clip(min=1e-5), size=len(cat_probs)
            ).clip(min=1e-5)
            cat_pis = cat_pis / cat_pis.sum(axis=1, keepdims=True)

            nexus_path = outprefix.with_suffix(".nex")
            f_str = write_nexus_profiles(cat_probs, cat_pis, nexus_path)
            cmd += ["-mdef", str(nexus_path)]
            model_str = f"GTR20{q_str}+FMIX{f_str}+G4{g_str}"
            alphabet = aa
            seq_type = "AA"

        else:
            raise ValueError(f"Invalid cfg.model: {cfg.model}")

        cmd += ["-m", model_str, "--seqtype", seq_type]

        if cfg.indels:
            ins_rate = cfg.hyper.max_ins * rng.uniform()
            del_rate = cfg.hyper.max_del * rng.uniform()
            cmd += ["--indel", f"{ins_rate},{del_rate}"]

        finished = False
        while not finished:
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                finished = True
            except subprocess.CalledProcessError:
                cmd[6] = str(int(rng.random() * 2**32))

        phy_file = outprefix.with_suffix(".phy")
        fa_file = outprefix.with_suffix(".unaligned.fa")

        if fa_file.exists() and cfg.realign:
            aln = list(SeqIO.parse(fa_file, "fasta"))
            taxa = [rec.id for rec in aln]
            seqs = np.array([str(rec.seq) for rec in aln])
            alignment = align(seqs, taxa)
            alignment = np.array([list(rec) for rec in alignment])
        else:
            aln = AlignIO.read(phy_file, "phylip-relaxed")
            taxa = [rec.id for rec in aln]
            alignment = np.array([list(rec.seq) for rec in aln])

        alignment = torch.tensor(np.vectorize(alphabet.index)(alignment))

        if cfg.freq_filter > 0:
            cols_to_keep = (alignment == len(alphabet) - 1).float().mean(
                dim=0
            ) < 1 - cfg.freq_filter
            alignment = alignment[:, cols_to_keep]

        if cfg.missing.mean > 0:
            a = cfg.missing.mean * cfg.missing.scale
            b = cfg.missing.scale - a
            p_missing = rng.beta(a, b)
            mask = rng.uniform(size=alignment.shape[:2]) < p_missing
            alignment[mask] = len(alphabet)

    return alignment, taxa
