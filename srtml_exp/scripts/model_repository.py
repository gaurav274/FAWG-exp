import click
import os


@click.command()
@click.option("--endpoint", type=str, default="http://localhost:4572")
def cleanmr(endpoint):
    os.system(
        f"aws s3 rm --endpoint {endpoint} s3://srtml-modellib --recursive"
    )


@click.command()
@click.option("--endpoint", type=str, default="http://localhost:4572")
def lsmr(endpoint):
    os.system(f"aws s3 ls --endpoint {endpoint} s3://srtml-modellib/")


@click.command()
@click.argument("s3_url")
@click.option("--endpoint", type=str, default="http://localhost:4572")
def plsmr(s3_url, endpoint):
    os.system(f"aws s3 ls --endpoint {endpoint} " + s3_url)


@click.command()
def start_srtml():

    # os.system(
    #     'ray start --head --resources \'{"Titan V": 8, "Tesla P40": 8, "Radeon '
    #     'RX 5000": 12, "Radeon RX Vega 64": 12, "CPU1": 8, "CPU2": 8, "FPGA1": '
    #     '4, "FPGA2": 4}\' --object-store-memory 1000000000 --num-cpus 24 '
    #     '--internal-config \'{"max_direct_call_object_size": 10485760, '
    #     '"max_grpc_message_size": 104857600}\''
    # )
    os.system(
        'ray start --head --resources \'{"Titan V": 8, "Tesla P40": 8, "Radeon '
        'RX 5000": 12, "Radeon RX Vega 64": 12, "CPU1": 8, "CPU2": 8, "FPGA1": '
        '4, "FPGA2": 4}\' --object-store-memory 100000000000 --num-cpus 24 '
    )
