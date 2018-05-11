
#http://click.pocoo.org/6/quickstart/#nesting-commands
import click
'''

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.obj['DEBUG'] = debug

@cli.command()
@click.pass_context
def sync(ctx):
    click.echo('Debug is %s' % (ctx.obj['DEBUG'] and 'on' or 'off'))


@click.command()
def download():
    click.echo('Download the database')
@click.add_command(download)

if __name__ == '__main__':
    cli(obj={})


'''


@click.group()
def cli():
	cli.add_command(download)
	cli.add_command(dropdb)
    #pass

@cli.command()
def download():
    click.echo('Initialized the download')
    import urllib2
    #url = 'https://www.dynaexamples.com/examples-manual/ls-dyna_example.zip'
    url = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    filename = url.split("?")[0].split("/")[-1]
    response = urllib2.urlopen(url) 
    zipcontent= response.read()    
    with open(filename, 'w') as f:
    	f.write(zipcontent)
    click.echo('finished the download')
    import zipfile
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('images')
    zip_ref.close()


@cli.command()
def dropdb():
    click.echo('Dropped the database')

if __name__ == '__main__':
    cli(obj={})