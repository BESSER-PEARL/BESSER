import uvicorn
import os, json
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from pydantic_classes import *
from sql_alchemy import *

############################################
#
#   Initialize the database
#
############################################

def init_db():
    SQLALCHEMY_DATABASE_URL = "sqlite:///./Domain Model.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal

app = FastAPI()

# Initialize database session
SessionLocal = init_db()
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

############################################
#
#   Reparation functions
#
############################################

@app.get("/reparation/", response_model=None)
def get_all_reparation(database: Session = Depends(get_db)) -> list[Reparation]:
    reparation_list = database.query(Reparation).all()
    return reparation_list


@app.get("/reparation/{reparation_id}/", response_model=None)
async def get_reparation(reparation_id: int, database: Session = Depends(get_db)) -> Reparation:
    db_reparation = database.query(Reparation).filter(Reparation.id == reparation_id).first()
    if db_reparation is None:
        raise HTTPException(status_code=404, detail="Reparation not found")

    response_data = {
        "reparation": db_reparation,
}
    return response_data



@app.post("/reparation/", response_model=None)
async def create_reparation(reparation_data: ReparationCreate, database: Session = Depends(get_db)) -> Reparation:

    if reparation_data.use_id is not None:
        db_use = database.query(Use).filter(Use.id == reparation_data.use_id).first()
        if not db_use:
            raise HTTPException(status_code=400, detail="Use not found")
    else:
        raise HTTPException(status_code=400, detail="Use ID is required")

    db_reparation = Reparation(date_set=reparation_data.date_set, description=reparation_data.description, use_id=reparation_data.use_id)

    database.add(db_reparation)
    database.commit()
    database.refresh(db_reparation)


    
    return db_reparation


@app.put("/reparation/{reparation_id}/", response_model=None)
async def update_reparation(reparation_id: int, reparation_data: ReparationCreate, database: Session = Depends(get_db)) -> Reparation:
    db_reparation = database.query(Reparation).filter(Reparation.id == reparation_id).first()
    if db_reparation is None:
        raise HTTPException(status_code=404, detail="Reparation not found")

    setattr(db_reparation, 'date_set', reparation_data.date_set)
    setattr(db_reparation, 'description', reparation_data.description)
    database.commit()
    database.refresh(db_reparation)
    return db_reparation


@app.delete("/reparation/{reparation_id}/", response_model=None)
async def delete_reparation(reparation_id: int, database: Session = Depends(get_db)):
    db_reparation = database.query(Reparation).filter(Reparation.id == reparation_id).first()
    if db_reparation is None:
        raise HTTPException(status_code=404, detail="Reparation not found")
    database.delete(db_reparation)
    database.commit()
    return db_reparation



############################################
#
#   LifecycleStage functions
#
############################################

@app.get("/lifecyclestage/", response_model=None)
def get_all_lifecyclestage(database: Session = Depends(get_db)) -> list[LifecycleStage]:
    lifecyclestage_list = database.query(LifecycleStage).all()
    return lifecyclestage_list


@app.get("/lifecyclestage/{lifecyclestage_id}/", response_model=None)
async def get_lifecyclestage(lifecyclestage_id: int, database: Session = Depends(get_db)) -> LifecycleStage:
    db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == lifecyclestage_id).first()
    if db_lifecyclestage is None:
        raise HTTPException(status_code=404, detail="LifecycleStage not found")

    rawmaterial_ids = database.query(composition.c.rawmaterial_id).filter(composition.c.lifecyclestage_id == db_lifecyclestage.id).all()
    productpassport_ids = database.query(stage.c.productpassport_id).filter(stage.c.lifecyclestage_id == db_lifecyclestage.id).all()
    response_data = {
        "lifecyclestage": db_lifecyclestage,
        "rawmaterial_ids": [x[0] for x in rawmaterial_ids],        "productpassport_ids": [x[0] for x in productpassport_ids]}
    return response_data



@app.post("/lifecyclestage/", response_model=None)
async def create_lifecyclestage(lifecyclestage_data: LifecycleStageCreate, database: Session = Depends(get_db)) -> LifecycleStage:


    db_lifecyclestage = LifecycleStage(start=lifecyclestage_data.start, end=lifecyclestage_data.end)

    database.add(db_lifecyclestage)
    database.commit()
    database.refresh(db_lifecyclestage)

    if lifecyclestage_data.rawmaterials_id:
        for id in lifecyclestage_data.rawmaterials_id:
            db_rawmaterial = database.query(RawMaterial).filter(RawMaterial.id == id).first()
            if not db_rawmaterial:
                raise HTTPException(status_code=404, detail=f"RawMaterial with ID {id} not found")
            # Create the association
            association = composition.insert().values(lifecyclestage_id=db_lifecyclestage.id, rawmaterial_id=db_rawmaterial.id)
            database.execute(association)
            database.commit()
    if lifecyclestage_data.productpassports_id:
        for id in lifecyclestage_data.productpassports_id:
            db_productpassport = database.query(ProductPassport).filter(ProductPassport.id == id).first()
            if not db_productpassport:
                raise HTTPException(status_code=404, detail=f"ProductPassport with ID {id} not found")
            # Create the association
            association = stage.insert().values(lifecyclestage_id=db_lifecyclestage.id, productpassport_id=db_productpassport.id)
            database.execute(association)
            database.commit()

    
    return db_lifecyclestage


@app.put("/lifecyclestage/{lifecyclestage_id}/", response_model=None)
async def update_lifecyclestage(lifecyclestage_id: int, lifecyclestage_data: LifecycleStageCreate, database: Session = Depends(get_db)) -> LifecycleStage:
    db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == lifecyclestage_id).first()
    if db_lifecyclestage is None:
        raise HTTPException(status_code=404, detail="LifecycleStage not found")

    setattr(db_lifecyclestage, 'start', lifecyclestage_data.start)
    setattr(db_lifecyclestage, 'end', lifecyclestage_data.end)
    existing_rawmaterial_ids = [assoc.rawmaterial_id for assoc in database.execute(
        composition.select().where(composition.c.lifecyclestage_id == db_lifecyclestage.id))]
    
    rawmaterials_to_remove = set(existing_rawmaterial_ids) - set(lifecyclestage_data.rawmaterials_id)
    for rawmaterial_id in rawmaterials_to_remove:
        association = composition.delete().where(
            composition.c.lifecyclestage_id == db_lifecyclestage.id and composition.c.rawmaterial_id == rawmaterial_id)
        database.execute(association)

    new_rawmaterial_ids = set(lifecyclestage_data.rawmaterials_id) - set(existing_rawmaterial_ids)
    for rawmaterial_id in new_rawmaterial_ids:
        db_rawmaterial = database.query(RawMaterial).filter(RawMaterial.id == rawmaterial_id).first()
        if db_rawmaterial is None:
            raise HTTPException(status_code=404, detail="RawMaterial with ID rawmaterial_id not found")
        association = composition.insert().values(lifecyclestage_id=db_lifecyclestage.id, rawmaterial_id=db_rawmaterial.id)
        database.execute(association)
    existing_productpassport_ids = [assoc.productpassport_id for assoc in database.execute(
        stage.select().where(stage.c.lifecyclestage_id == db_lifecyclestage.id))]
    
    productpassports_to_remove = set(existing_productpassport_ids) - set(lifecyclestage_data.productpassports_id)
    for productpassport_id in productpassports_to_remove:
        association = stage.delete().where(
            stage.c.lifecyclestage_id == db_lifecyclestage.id and stage.c.productpassport_id == productpassport_id)
        database.execute(association)

    new_productpassport_ids = set(lifecyclestage_data.productpassports_id) - set(existing_productpassport_ids)
    for productpassport_id in new_productpassport_ids:
        db_productpassport = database.query(ProductPassport).filter(ProductPassport.id == productpassport_id).first()
        if db_productpassport is None:
            raise HTTPException(status_code=404, detail="ProductPassport with ID productpassport_id not found")
        association = stage.insert().values(lifecyclestage_id=db_lifecyclestage.id, productpassport_id=db_productpassport.id)
        database.execute(association)
    database.commit()
    database.refresh(db_lifecyclestage)
    return db_lifecyclestage


@app.delete("/lifecyclestage/{lifecyclestage_id}/", response_model=None)
async def delete_lifecyclestage(lifecyclestage_id: int, database: Session = Depends(get_db)):
    db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == lifecyclestage_id).first()
    if db_lifecyclestage is None:
        raise HTTPException(status_code=404, detail="LifecycleStage not found")
    database.delete(db_lifecyclestage)
    database.commit()
    return db_lifecyclestage



############################################
#
#   Design functions
#
############################################

@app.get("/design/", response_model=None)
def get_all_design(database: Session = Depends(get_db)) -> list[Design]:
    design_list = database.query(Design).all()
    return design_list


@app.get("/design/{design_id}/", response_model=None)
async def get_design(design_id: int, database: Session = Depends(get_db)) -> Design:
    db_design = database.query(Design).filter(Design.id == design_id).first()
    if db_design is None:
        raise HTTPException(status_code=404, detail="Design not found")

    response_data = {
        "design": db_design,
}
    return response_data



@app.post("/design/", response_model=None)
async def create_design(design_data: DesignCreate, database: Session = Depends(get_db)) -> Design:


    db_design = Design(start=design_data.start, end=design_data.end)

    database.add(db_design)
    database.commit()
    database.refresh(db_design)


    
    return db_design


@app.put("/design/{design_id}/", response_model=None)
async def update_design(design_id: int, design_data: DesignCreate, database: Session = Depends(get_db)) -> Design:
    db_design = database.query(Design).filter(Design.id == design_id).first()
    if db_design is None:
        raise HTTPException(status_code=404, detail="Design not found")

    database.commit()
    database.refresh(db_design)
    return db_design


@app.delete("/design/{design_id}/", response_model=None)
async def delete_design(design_id: int, database: Session = Depends(get_db)):
    db_design = database.query(Design).filter(Design.id == design_id).first()
    if db_design is None:
        raise HTTPException(status_code=404, detail="Design not found")
    database.delete(db_design)
    database.commit()
    return db_design



############################################
#
#   Use functions
#
############################################

@app.get("/use/", response_model=None)
def get_all_use(database: Session = Depends(get_db)) -> list[Use]:
    use_list = database.query(Use).all()
    return use_list


@app.get("/use/{use_id}/", response_model=None)
async def get_use(use_id: int, database: Session = Depends(get_db)) -> Use:
    db_use = database.query(Use).filter(Use.id == use_id).first()
    if db_use is None:
        raise HTTPException(status_code=404, detail="Use not found")

    response_data = {
        "use": db_use,
}
    return response_data



@app.post("/use/", response_model=None)
async def create_use(use_data: UseCreate, database: Session = Depends(get_db)) -> Use:


    db_use = Use(start=use_data.start, end=use_data.end)

    database.add(db_use)
    database.commit()
    database.refresh(db_use)


    
    return db_use


@app.put("/use/{use_id}/", response_model=None)
async def update_use(use_id: int, use_data: UseCreate, database: Session = Depends(get_db)) -> Use:
    db_use = database.query(Use).filter(Use.id == use_id).first()
    if db_use is None:
        raise HTTPException(status_code=404, detail="Use not found")

    database.commit()
    database.refresh(db_use)
    return db_use


@app.delete("/use/{use_id}/", response_model=None)
async def delete_use(use_id: int, database: Session = Depends(get_db)):
    db_use = database.query(Use).filter(Use.id == use_id).first()
    if db_use is None:
        raise HTTPException(status_code=404, detail="Use not found")
    database.delete(db_use)
    database.commit()
    return db_use



############################################
#
#   Manufacture functions
#
############################################

@app.get("/manufacture/", response_model=None)
def get_all_manufacture(database: Session = Depends(get_db)) -> list[Manufacture]:
    manufacture_list = database.query(Manufacture).all()
    return manufacture_list


@app.get("/manufacture/{manufacture_id}/", response_model=None)
async def get_manufacture(manufacture_id: int, database: Session = Depends(get_db)) -> Manufacture:
    db_manufacture = database.query(Manufacture).filter(Manufacture.id == manufacture_id).first()
    if db_manufacture is None:
        raise HTTPException(status_code=404, detail="Manufacture not found")

    response_data = {
        "manufacture": db_manufacture,
}
    return response_data



@app.post("/manufacture/", response_model=None)
async def create_manufacture(manufacture_data: ManufactureCreate, database: Session = Depends(get_db)) -> Manufacture:


    db_manufacture = Manufacture(start=manufacture_data.start, end=manufacture_data.end)

    database.add(db_manufacture)
    database.commit()
    database.refresh(db_manufacture)


    
    return db_manufacture


@app.put("/manufacture/{manufacture_id}/", response_model=None)
async def update_manufacture(manufacture_id: int, manufacture_data: ManufactureCreate, database: Session = Depends(get_db)) -> Manufacture:
    db_manufacture = database.query(Manufacture).filter(Manufacture.id == manufacture_id).first()
    if db_manufacture is None:
        raise HTTPException(status_code=404, detail="Manufacture not found")

    database.commit()
    database.refresh(db_manufacture)
    return db_manufacture


@app.delete("/manufacture/{manufacture_id}/", response_model=None)
async def delete_manufacture(manufacture_id: int, database: Session = Depends(get_db)):
    db_manufacture = database.query(Manufacture).filter(Manufacture.id == manufacture_id).first()
    if db_manufacture is None:
        raise HTTPException(status_code=404, detail="Manufacture not found")
    database.delete(db_manufacture)
    database.commit()
    return db_manufacture



############################################
#
#   Distribution functions
#
############################################

@app.get("/distribution/", response_model=None)
def get_all_distribution(database: Session = Depends(get_db)) -> list[Distribution]:
    distribution_list = database.query(Distribution).all()
    return distribution_list


@app.get("/distribution/{distribution_id}/", response_model=None)
async def get_distribution(distribution_id: int, database: Session = Depends(get_db)) -> Distribution:
    db_distribution = database.query(Distribution).filter(Distribution.id == distribution_id).first()
    if db_distribution is None:
        raise HTTPException(status_code=404, detail="Distribution not found")

    response_data = {
        "distribution": db_distribution,
}
    return response_data



@app.post("/distribution/", response_model=None)
async def create_distribution(distribution_data: DistributionCreate, database: Session = Depends(get_db)) -> Distribution:


    db_distribution = Distribution(start=distribution_data.start, end=distribution_data.end)

    database.add(db_distribution)
    database.commit()
    database.refresh(db_distribution)


    
    return db_distribution


@app.put("/distribution/{distribution_id}/", response_model=None)
async def update_distribution(distribution_id: int, distribution_data: DistributionCreate, database: Session = Depends(get_db)) -> Distribution:
    db_distribution = database.query(Distribution).filter(Distribution.id == distribution_id).first()
    if db_distribution is None:
        raise HTTPException(status_code=404, detail="Distribution not found")

    database.commit()
    database.refresh(db_distribution)
    return db_distribution


@app.delete("/distribution/{distribution_id}/", response_model=None)
async def delete_distribution(distribution_id: int, database: Session = Depends(get_db)):
    db_distribution = database.query(Distribution).filter(Distribution.id == distribution_id).first()
    if db_distribution is None:
        raise HTTPException(status_code=404, detail="Distribution not found")
    database.delete(db_distribution)
    database.commit()
    return db_distribution



############################################
#
#   RawMaterial functions
#
############################################

@app.get("/rawmaterial/", response_model=None)
def get_all_rawmaterial(database: Session = Depends(get_db)) -> list[RawMaterial]:
    rawmaterial_list = database.query(RawMaterial).all()
    return rawmaterial_list


@app.get("/rawmaterial/{rawmaterial_id}/", response_model=None)
async def get_rawmaterial(rawmaterial_id: int, database: Session = Depends(get_db)) -> RawMaterial:
    db_rawmaterial = database.query(RawMaterial).filter(RawMaterial.id == rawmaterial_id).first()
    if db_rawmaterial is None:
        raise HTTPException(status_code=404, detail="RawMaterial not found")

    lifecyclestage_ids = database.query(composition.c.lifecyclestage_id).filter(composition.c.rawmaterial_id == db_rawmaterial.id).all()
    response_data = {
        "rawmaterial": db_rawmaterial,
        "lifecyclestage_ids": [x[0] for x in lifecyclestage_ids]}
    return response_data



@app.post("/rawmaterial/", response_model=None)
async def create_rawmaterial(rawmaterial_data: RawMaterialCreate, database: Session = Depends(get_db)) -> RawMaterial:


    db_rawmaterial = RawMaterial(name=rawmaterial_data.name)

    database.add(db_rawmaterial)
    database.commit()
    database.refresh(db_rawmaterial)

    if rawmaterial_data.lifecyclestages_id:
        for id in rawmaterial_data.lifecyclestages_id:
            db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == id).first()
            if not db_lifecyclestage:
                raise HTTPException(status_code=404, detail=f"LifecycleStage with ID {id} not found")
            # Create the association
            association = composition.insert().values(rawmaterial_id=db_rawmaterial.id, lifecyclestage_id=db_lifecyclestage.id)
            database.execute(association)
            database.commit()

    
    return db_rawmaterial


@app.put("/rawmaterial/{rawmaterial_id}/", response_model=None)
async def update_rawmaterial(rawmaterial_id: int, rawmaterial_data: RawMaterialCreate, database: Session = Depends(get_db)) -> RawMaterial:
    db_rawmaterial = database.query(RawMaterial).filter(RawMaterial.id == rawmaterial_id).first()
    if db_rawmaterial is None:
        raise HTTPException(status_code=404, detail="RawMaterial not found")

    setattr(db_rawmaterial, 'name', rawmaterial_data.name)
    existing_lifecyclestage_ids = [assoc.lifecyclestage_id for assoc in database.execute(
        composition.select().where(composition.c.rawmaterial_id == db_rawmaterial.id))]
    
    lifecyclestages_to_remove = set(existing_lifecyclestage_ids) - set(rawmaterial_data.lifecyclestages_id)
    for lifecyclestage_id in lifecyclestages_to_remove:
        association = composition.delete().where(
            composition.c.rawmaterial_id == db_rawmaterial.id and composition.c.lifecyclestage_id == lifecyclestage_id)
        database.execute(association)

    new_lifecyclestage_ids = set(rawmaterial_data.lifecyclestages_id) - set(existing_lifecyclestage_ids)
    for lifecyclestage_id in new_lifecyclestage_ids:
        db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == lifecyclestage_id).first()
        if db_lifecyclestage is None:
            raise HTTPException(status_code=404, detail="LifecycleStage with ID lifecyclestage_id not found")
        association = composition.insert().values(rawmaterial_id=db_rawmaterial.id, lifecyclestage_id=db_lifecyclestage.id)
        database.execute(association)
    database.commit()
    database.refresh(db_rawmaterial)
    return db_rawmaterial


@app.delete("/rawmaterial/{rawmaterial_id}/", response_model=None)
async def delete_rawmaterial(rawmaterial_id: int, database: Session = Depends(get_db)):
    db_rawmaterial = database.query(RawMaterial).filter(RawMaterial.id == rawmaterial_id).first()
    if db_rawmaterial is None:
        raise HTTPException(status_code=404, detail="RawMaterial not found")
    database.delete(db_rawmaterial)
    database.commit()
    return db_rawmaterial



############################################
#
#   ProductPassport functions
#
############################################

@app.get("/productpassport/", response_model=None)
def get_all_productpassport(database: Session = Depends(get_db)) -> list[ProductPassport]:
    productpassport_list = database.query(ProductPassport).all()
    return productpassport_list


@app.get("/productpassport/{productpassport_id}/", response_model=None)
async def get_productpassport(productpassport_id: int, database: Session = Depends(get_db)) -> ProductPassport:
    db_productpassport = database.query(ProductPassport).filter(ProductPassport.id == productpassport_id).first()
    if db_productpassport is None:
        raise HTTPException(status_code=404, detail="ProductPassport not found")

    lifecyclestage_ids = database.query(stage.c.lifecyclestage_id).filter(stage.c.productpassport_id == db_productpassport.id).all()
    response_data = {
        "productpassport": db_productpassport,
        "lifecyclestage_ids": [x[0] for x in lifecyclestage_ids]}
    return response_data



@app.post("/productpassport/", response_model=None)
async def create_productpassport(productpassport_data: ProductPassportCreate, database: Session = Depends(get_db)) -> ProductPassport:


    db_productpassport = ProductPassport(product_name=productpassport_data.product_name, brand=productpassport_data.brand, code=productpassport_data.code)

    database.add(db_productpassport)
    database.commit()
    database.refresh(db_productpassport)

    if productpassport_data.lifecyclestages_id:
        for id in productpassport_data.lifecyclestages_id:
            db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == id).first()
            if not db_lifecyclestage:
                raise HTTPException(status_code=404, detail=f"LifecycleStage with ID {id} not found")
            # Create the association
            association = stage.insert().values(productpassport_id=db_productpassport.id, lifecyclestage_id=db_lifecyclestage.id)
            database.execute(association)
            database.commit()

    
    return db_productpassport


@app.put("/productpassport/{productpassport_id}/", response_model=None)
async def update_productpassport(productpassport_id: int, productpassport_data: ProductPassportCreate, database: Session = Depends(get_db)) -> ProductPassport:
    db_productpassport = database.query(ProductPassport).filter(ProductPassport.id == productpassport_id).first()
    if db_productpassport is None:
        raise HTTPException(status_code=404, detail="ProductPassport not found")

    setattr(db_productpassport, 'product_name', productpassport_data.product_name)
    setattr(db_productpassport, 'brand', productpassport_data.brand)
    setattr(db_productpassport, 'code', productpassport_data.code)
    existing_lifecyclestage_ids = [assoc.lifecyclestage_id for assoc in database.execute(
        stage.select().where(stage.c.productpassport_id == db_productpassport.id))]
    
    lifecyclestages_to_remove = set(existing_lifecyclestage_ids) - set(productpassport_data.lifecyclestages_id)
    for lifecyclestage_id in lifecyclestages_to_remove:
        association = stage.delete().where(
            stage.c.productpassport_id == db_productpassport.id and stage.c.lifecyclestage_id == lifecyclestage_id)
        database.execute(association)

    new_lifecyclestage_ids = set(productpassport_data.lifecyclestages_id) - set(existing_lifecyclestage_ids)
    for lifecyclestage_id in new_lifecyclestage_ids:
        db_lifecyclestage = database.query(LifecycleStage).filter(LifecycleStage.id == lifecyclestage_id).first()
        if db_lifecyclestage is None:
            raise HTTPException(status_code=404, detail="LifecycleStage with ID lifecyclestage_id not found")
        association = stage.insert().values(productpassport_id=db_productpassport.id, lifecyclestage_id=db_lifecyclestage.id)
        database.execute(association)
    database.commit()
    database.refresh(db_productpassport)
    return db_productpassport


@app.delete("/productpassport/{productpassport_id}/", response_model=None)
async def delete_productpassport(productpassport_id: int, database: Session = Depends(get_db)):
    db_productpassport = database.query(ProductPassport).filter(ProductPassport.id == productpassport_id).first()
    if db_productpassport is None:
        raise HTTPException(status_code=404, detail="ProductPassport not found")
    database.delete(db_productpassport)
    database.commit()
    return db_productpassport





############################################
# Maintaining the server
############################################
if __name__ == "__main__":
    import uvicorn
    openapi_schema = app.openapi()
    output_dir = os.path.join(os.getcwd(), 'output_backend')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'openapi_specs.json')
    print(f"Writing OpenAPI schema to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(openapi_schema, file)
    uvicorn.run(app, host="0.0.0.0", port=8000)



