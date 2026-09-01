# Split argos_plugin/tests/test_hybrid_memory.py into per-class domain files (issue #98).
# One file per test class, identical header (imports + sys.path setup) prepended to each.
# The unused module-level `tmp_path_factory_dir` fixture is dropped (provably unused: only
# one match in the monolith = its own definition).
[CmdletBinding()]
param(
    [string] $TestsDir = "argos_plugin/tests"
)

$ErrorActionPreference = "Stop"
$src = Join-Path $TestsDir "test_hybrid_memory.py"
$lines = Get-Content $src

# Header = lines 1..22 (1-indexed) = indices 0..21
$header = $lines[0..21]

# Class ranges: (ClassName, startLine1Indexed, endLine1Indexed, fileName)
# end = next class start - 1; last class end = last content line (3889).
$classes = @(
    @{ Name="TestEmbeddings";                 Start=31;   End=119;  File="test_hybrid_embeddings.py" },
    @{ Name="TestDuckDBStore";                Start=120;  End=1045; File="test_hybrid_duckdb_store.py" },
    @{ Name="TestHybridRanking";              Start=1046; End=1189; File="test_hybrid_ranking.py" },
    @{ Name="TestSharedStoreSurface";         Start=1190; End=1322; File="test_hybrid_shared_store.py" },
    @{ Name="TestExtractionAndDedup";         Start=1323; End=1420; File="test_hybrid_extraction_dedup.py" },
    @{ Name="TestInsightLog";                 Start=1421; End=1529; File="test_hybrid_insight_log.py" },
    @{ Name="TestPriority3GraphEnhancements"; Start=1530; End=2178; File="test_hybrid_graph_enhancements.py" },
    @{ Name="TestCandidateQueue";             Start=2179; End=2293; File="test_hybrid_candidate_queue.py" },
    @{ Name="TestProviderInit";               Start=2294; End=2327; File="test_hybrid_provider_init.py" },
    @{ Name="TestKuzuGraph";                  Start=2328; End=2565; File="test_hybrid_kuzu_graph.py" },
    @{ Name="TestExtractor";                  Start=2566; End=2617; File="test_hybrid_extractor.py" },
    @{ Name="TestStorageRouting";             Start=2618; End=2634; File="test_hybrid_storage_routing.py" },
    @{ Name="TestPluginDiscovery";            Start=2635; End=2651; File="test_hybrid_plugin_discovery.py" },
    @{ Name="TestMemoryUpdateProviderPath";   Start=2652; End=3371; File="test_hybrid_memory_update_provider.py" },
    @{ Name="TestEvolutionChains";            Start=3372; End=3889; File="test_hybrid_evolution_chains.py" }
)

$totalTests = 0
foreach ($c in $classes) {
    # Convert 1-indexed inclusive range to 0-indexed array slice.
    $body = $lines[($c.Start - 1)..($c.End - 1)]
    $out = Join-Path $TestsDir $c.File
    # 2 blank lines between header and class (PEP 8); original had the fixture there.
    $content = $header + @("", "") + $body
    # Write LF line endings (repo .gitattributes enforces eol=lf for .py) with UTF-8
    # BOM to match the original monolith's encoding. Set-Content would emit CRLF.
    $text = ($content -join "`n") + "`n"
    [System.IO.File]::WriteAllText($out, $text, [System.Text.UTF8Encoding]::new($true))
    # Count tests in this class for a sanity report.
    $count = ($body | Select-String -Pattern '^\s+def test_').Count
    $totalTests += $count
    Write-Output ("{0,-45} lines {1,5}-{2,5}  tests={3,3}  -> {4}" -f $c.Name, $c.Start, $c.End, $count, $c.File)
}
Write-Output ("TOTAL tests across split files: $totalTests")
