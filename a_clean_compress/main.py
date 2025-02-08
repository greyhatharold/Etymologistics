"""
Main entry point for the data processing pipeline.
Provides an interactive command-line interface for configuring and running the pipeline.
"""

import sys
from pathlib import Path
from typing import Optional
import click
from click_help_colors import HelpColorsGroup, HelpColorsCommand
from click_option_group import optgroup, OptionGroup, MutuallyExclusiveOptionGroup

from logger_config import setup_logging, LogConfigError
from pipeline_manager import run_pipeline, PipelineError


def create_pipeline_config(
    input_path: str,
    output_path: str,
    compression_method: str,
    compression_level: Optional[int],
    date_columns: Optional[str],
    numeric_columns: Optional[str],
    output_format: str = 'csv'
) -> dict:
    """
    Create pipeline configuration dictionary from CLI parameters.
    
    Args:
        input_path: Path to input data file or URL
        output_path: Path for output file
        compression_method: Compression algorithm to use
        compression_level: Optional compression level
        date_columns: Comma-separated list of date column names
        numeric_columns: Comma-separated list of numeric column names
        output_format: Output format for the processed data
        
    Returns:
        Configuration dictionary for pipeline
    """
    config = {
        'input_path': input_path,
        'output_path': output_path,
        'compression_method': compression_method,
        'output_format': output_format
    }
    
    if compression_level is not None:
        config['compression_level'] = compression_level
    
    # Create cleaning configuration if specified
    cleaning_config = {}
    if date_columns:
        cleaning_config['date_columns'] = date_columns.split(',')
    if numeric_columns:
        cleaning_config['numeric_columns'] = numeric_columns.split(',')
    
    if cleaning_config:
        config['cleaning_config'] = cleaning_config
    
    return config


class ColorGroup(HelpColorsGroup):
    def get_help(self, ctx):
        """Override to add custom formatting to help text."""
        return click.style("""
╭────────────────────────────────────────────╮
│     Data Processing Pipeline CLI Tool      │
╰────────────────────────────────────────────╯
        """, fg='blue') + super().get_help(ctx)


@click.group(
    cls=ColorGroup,
    help_headers_color='yellow',
    help_options_color='green'
)
def cli():
    """
    Data Processing Pipeline CLI Tool

    This tool processes data through three stages:
    1. Data ingestion (from local file or URL)
    2. Data cleaning (with customizable options)
    3. Data compression (with multiple algorithms)

    Examples:

    \b
    Basic usage:
        $ python main.py process -i data/input.csv -o data/output.csv

    \b
    With specific compression:
        $ python main.py process -i data/input.csv -o data/output.csv -c bz2 -l 9

    \b
    With data cleaning options:
        $ python main.py process -i data/input.csv -o data/output.csv \\
            --date-columns "date,timestamp" --numeric-columns "value,count"

    \b
    With logging configuration:
        $ python main.py process -i data/input.csv -o data/output.csv \\
            --log-level DEBUG --log-file logs/pipeline.log
    """
    pass


@cli.command(
    cls=HelpColorsCommand,
    help_headers_color='yellow',
    help_options_color='green'
)
@optgroup.group('Input/Output', cls=OptionGroup)
@optgroup.option(
    '--input-path', '-i',
    type=click.Path(exists=True),
    required=True,
    help='📄 Path to input file (supports .csv, .json, .xlsx, .torrent) or magnet link'
)
@optgroup.option(
    '--output-path', '-o',
    type=click.Path(),
    required=True,
    help='💾 Path for output file'
)
@optgroup.option(
    '--output-format', '-f',
    type=click.Choice(['csv', 'json']),
    default='csv',
    help='📊 Output format for the processed data'
)
@optgroup.group('Compression Options')
@optgroup.option(
    '--compression', '-c',
    type=click.Choice(['gzip', 'bz2', 'lzma'], case_sensitive=False),
    default='gzip',
    help='🗜️  Compression method to use'
)
@optgroup.option(
    '--compression-level', '-l',
    type=click.IntRange(1, 9),
    help='📊 Compression level (1-9, higher = better compression)'
)
@optgroup.group('Data Cleaning Options')
@optgroup.option(
    '--date-columns',
    help='📅 Comma-separated list of date column names'
)
@optgroup.option(
    '--numeric-columns',
    help='🔢 Comma-separated list of numeric column names'
)
@optgroup.group('Logging Options')
@optgroup.option(
    '--log-level',
    type=click.Choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        case_sensitive=False
    ),
    default='INFO',
    help='📝 Logging level'
)
@optgroup.option(
    '--log-file',
    type=click.Path(),
    help='📁 Path to log file (optional)'
)
def process(
    input_path: str,
    output_path: str,
    compression: str,
    compression_level: Optional[int],
    date_columns: Optional[str],
    numeric_columns: Optional[str],
    output_format: str,
    log_level: str,
    log_file: Optional[str]
) -> None:
    """
    Process data through the pipeline.
    
    This command will:
    1. Load data from the specified input source
    2. Clean the data (if cleaning options are provided)
    3. Compress the result using the specified method
    """
    try:
        # Setup logging
        logger = setup_logging(
            level=log_level,
            log_file=log_file,
            module_name="pipeline"
        )
        
        logger.info("Starting data processing pipeline")
        logger.debug(f"Input path: {input_path}")
        logger.debug(f"Output path: {output_path}")
        logger.debug(f"Output format: {output_format}")
        logger.debug(f"Compression method: {compression}")
        
        # Show progress
        with click.progressbar(
            length=3,
            label=click.style('Processing', fg='green'),
            fill_char=click.style('█', fg='green'),
            empty_char='░'
        ) as bar:
            # Create pipeline configuration
            config = create_pipeline_config(
                input_path=input_path,
                output_path=output_path,
                compression_method=compression,
                compression_level=compression_level,
                date_columns=date_columns,
                numeric_columns=numeric_columns,
                output_format=output_format
            )
            bar.update(1)
            
            # Run pipeline
            output_path = run_pipeline(config)
            bar.update(1)
            
            # Final update
            bar.update(1)
        
        click.echo(click.style(
            f"\n✨ Pipeline completed successfully!\n"
            f"📦 Output saved to: {output_path}",
            fg='green'
        ))
        sys.exit(0)
        
    except (LogConfigError, PipelineError) as e:
        click.echo(click.style(f"\n❌ Pipeline failed: {str(e)}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"\n💥 Unexpected error: {str(e)}", fg='red'))
        sys.exit(2)


@cli.command()
def version():
    """Show the version of the pipeline tool."""
    click.echo(click.style("Data Processing Pipeline v1.0.0", fg='blue'))


if __name__ == '__main__':
    cli() 