$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pptPath = Join-Path $projectRoot 'H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pptx'
$pdfPath = [IO.Path]::ChangeExtension($pptPath, '.pdf')
$pdfBackup = Join-Path $projectRoot 'results/verified_20260913/original_ppt_backup/H&E空间转录组生成研究_统一Benchmark汇报_庹力元.pdf'
if ((Test-Path -LiteralPath $pdfPath) -and -not (Test-Path -LiteralPath $pdfBackup)) {
    Copy-Item -LiteralPath $pdfPath -Destination $pdfBackup
}
$powerpointApp = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    # ReadOnly=True, Untitled=False, WithWindow=False: no visible editor window.
    $deck = $powerpointApp.Presentations.Open($pptPath, -1, 0, 0)
    if ($deck.Slides.Count -ne 11) { throw "Unexpected slide count: $($deck.Slides.Count)" }
    # ppSaveAsPDF=32 avoids the optional-argument COM binding issue in pwsh.
    $deck.SaveAs($pdfPath, 32)
    $overflow = @()
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $contentHeight = $shape.Height - $shape.TextFrame.MarginTop - $shape.TextFrame.MarginBottom
                $boundHeight = $shape.TextFrame.TextRange.BoundHeight
                if ($boundHeight -gt $contentHeight + 3) {
                    $overflow += [pscustomobject]@{ Slide = $slide.SlideIndex; Shape = $shape.Name; BoundHeight = $boundHeight; Available = $contentHeight; Text = $shape.TextFrame.TextRange.Text }
                }
            }
        }
    }
    $result = @{ PDF = $pdfPath; Slides = $deck.Slides.Count; TextOverflow = @($overflow) }
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $projectRoot 'results/verified_20260913/powerpoint_render_audit.json') -Encoding utf8
    $result | ConvertTo-Json -Depth 5
} finally {
    if ($null -ne $deck) { $deck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    if ($powerpointApp.Presentations.Count -eq 0) { $powerpointApp.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerpointApp)
}
